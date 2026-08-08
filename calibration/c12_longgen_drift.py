#!/usr/bin/env python3
"""C-12: run the drift mechanism against the real long-generation curve.

Declared in ``DECLARATION-C12.md``, sealed at commit 90e2ce2 before this ran.

The setup is what makes the test possible. ``tq_paper_lb_shard.py`` quantizes
each layer's settled prefill **once**, guarded by a ``_qdone`` set, and every
generated token is concatenated in fp16. So a single fixed perturbation drives
the entire 512-token generation and the damage still grows. Two mechanisms
compete for that: the repo's recorded autoregressive compounding, and C-11c's
operator drift.

Teacher forcing separates them. Feed both arms the identical fp16-generated
continuation and the compounding channel is closed by construction, leaving
the frozen key error as the only difference between them.

Quantization is **imported from the harness**, not re-implemented, so the
perturbation is bit-identical to the run that produced the published curve.
The read operator is **imported from c11c_operator_drift**, so the geometry is
the same estimator C-11c validated against its null.

Runs on Atlas, GPU 1, kvbench venv.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---- harness config, matched to tq_abl.sh, set BEFORE importing the harness --
HARNESS_ENV = {
    "CODEBOOK": "nf4a",
    "KEY_BITS": "4",
    "VAL_BITS": "4",
    "GROUP": "32",
    "HOT": "128",
    "SINK": "4",
    "OUTLIER_FRAC": "0.02",
    "PREROPE": "0",
    "NOQUANT": "0",
    "SHARD_ID": "0",
    "NUM_SHARDS": "1",
    "MODEL": "Qwen/Qwen2.5-7B-Instruct",
    "MODEL_KEY": "qwen2.5-7b-instruct",
}
os.environ.update(HARNESS_ENV)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import (  # noqa: E402
    apply_rotary_pos_emb,
)

HARNESS_DIR = os.environ.get("HARNESS_DIR", "/archive/c12")
sys.path.insert(0, HARNESS_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tq_paper_lb_shard as HARNESS  # noqa: E402

# The harness patches DynamicCache.update at import so its own runs quantize
# implicitly. Undo it: this experiment quantizes the caches it chooses,
# explicitly, so the fp16 arm stays exactly fp16.
DynamicCache.update = HARNESS._orig_update

import c11c_operator_drift as C11C  # noqa: E402

# ---- declared constants, from DECLARATION-C12.md ----------------------------
N_DOCS = int(os.environ.get("N_DOCS", "40"))
LAYERS = [4, 14, 24]
MAXGEN = 512
EARLY = (0, 128)
LATE = (384, 512)
N_WINDOWS = 4
N_ROT_DRAWS = 5
PROBE_KEYS = 24
SEED = 20260808
A0_ROUGE_BAR = 5.0
A1_NATS_BAR = 0.01
B1_GROWTH_BAR = 0.05
B2_ALPHA = 0.05

MAXLEN = 31500
LB_DATA = "/archive/longbench/data/gov_report.jsonl"
LB_CONF = "/archive/longbench/config"
OUTDIR = Path(os.environ.get("OUTDIR", "/archive/c12/out"))

try:
    from rouge_score import rouge_scorer

    _RS = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rouge_l(pred, refs):
        return 100.0 * max(
            _RS.score(r, pred)["rougeL"].fmeasure for r in refs
        )

    ROUGE_IMPL = "rouge_score.RougeScorer(rougeL, stemmer)"
except Exception:  # pragma: no cover - recorded in the record either way

    def _lcs(a, b):
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
            prev = cur
        return prev[-1]

    def rouge_l(pred, refs):
        p = pred.lower().split()
        best = 0.0
        for r in refs:
            g = r.lower().split()
            if not p or not g:
                continue
            m = _lcs(p, g)
            if m:
                prec, rec = m / len(p), m / len(g)
                best = max(best, 2 * prec * rec / (prec + rec))
        return 100.0 * best

    ROUGE_IMPL = "fallback whitespace LCS F1 (rouge_score unavailable)"


def spearman(x, y):
    """Spearman rho and a two-sided p from the t approximation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 4:
        return 0.0, 1.0

    def rank(v):
        order = v.argsort()
        r = np.empty(n, float)
        r[order] = np.arange(n, dtype=float)
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        if cnt.max() > 1:
            sums = np.zeros(len(cnt))
            np.add.at(sums, inv, r)
            r = (sums / cnt)[inv]
        return r

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho**2))
    # two-sided p from a t distribution with n-2 df, via the incomplete beta
    from math import lgamma, log

    df = n - 2

    def _betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = np.exp(a * log(x) + b * log(1 - x) - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 300):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1.0 - c * d) < 1e-10:
                break
        return front * (f - 1.0)

    xx = df / (df + t * t)
    p = _betai(df / 2.0, 0.5, xx)
    return rho, float(min(1.0, max(0.0, p)))


def sign_test_p(vals, bar=0.0):
    """Two-sided sign test that values exceed `bar`, exact binomial."""
    from math import comb

    pos = sum(1 for v in vals if v > bar)
    neg = sum(1 for v in vals if v < bar)
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def random_orthogonal(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))[None, :]


def cache_keys(cache, li):
    if hasattr(cache, "key_cache"):
        return cache.key_cache[li], cache.value_cache[li]
    lay = cache.layers[li]
    return lay.keys, lay.values


def n_cache_layers(cache):
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache.layers)


def quantize_prefill(cache, hot):
    """Exactly what _patched_update does on the first update of each layer."""
    for li in range(n_cache_layers(cache)):
        fk, fv = cache_keys(cache, li)
        T = fk.shape[2]
        n = max(0, T - hot)
        if n > 0:
            HARNESS._CUR_LAYER = li
            fk[:, :, :n, :] = HARNESS.qdq_key_block(fk[:, :, :n, :])
            fv[:, :, :n, :] = HARNESS.qdq_val_block(fv[:, :, :n, :])
    return cache


def build_prompt(tok, rec, prompt_fmt):
    prompt = prompt_fmt.format(**rec)
    tp = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tp) > MAXLEN:
        h = MAXLEN // 2
        prompt = tok.decode(tp[:h], skip_special_tokens=True) + tok.decode(
            tp[-h:], skip_special_tokens=True
        )
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


class QueryTap:
    """Capture pre-RoPE queries on the declared layers, then rotate them.

    Hooking q_proj gives pre-RoPE queries; the rotation is applied here with
    the position ids this experiment already knows exactly, so nothing about
    the model's internal rope plumbing has to be guessed at.
    """

    def __init__(self, model, layers, n_heads, head_dim):
        self.model, self.layers = model, layers
        self.n_heads, self.head_dim = n_heads, head_dim
        self.buf, self.handles, self.on = {}, [], False
        for li in layers:
            mod = model.model.layers[li].self_attn.q_proj
            self.handles.append(
                mod.register_forward_hook(self._mk(li))
            )

    def _mk(self, li):
        def hook(_m, _i, out):
            if self.on:
                self.buf[li] = out.detach()

        return hook

    def rotated(self, li, position_ids, ref):
        q = self.buf[li]
        B, S, _ = q.shape
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.model.model.rotary_emb(ref, position_ids)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        return q_rot

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def greedy(model, tok, cache, first_logits, n_new, device):
    """Greedy decode from a prepared cache. Returns tokens and their NLL."""
    toks, nlls = [], []
    logits = first_logits
    T = cache_keys(cache, 0)[0].shape[2]
    for i in range(n_new):
        lp = torch.log_softmax(logits.float(), dim=-1)
        nt = int(lp.argmax(-1))
        toks.append(nt)
        nlls.append(-float(lp[nt]))
        if nt == tok.eos_token_id:
            break
        pos = torch.tensor([[T + i]], device=device)
        out = model(
            input_ids=torch.tensor([[nt]], device=device),
            past_key_values=cache,
            position_ids=pos,
            use_cache=True,
        )
        logits = out.logits[0, -1]
    return toks, nlls


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    prompt_fmt = json.load(open(f"{LB_CONF}/dataset2prompt.json"))["gov_report"]
    data = [json.loads(x) for x in open(LB_DATA)][:N_DOCS]

    tok = AutoTokenizer.from_pretrained(HARNESS_ENV["MODEL"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        HARNESS_ENV["MODEL"],
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()
    cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    grp = n_q // n_kv
    print(
        f"[c12] layers={cfg.num_hidden_layers} q={n_q} kv={n_kv} hd={hd} "
        f"grp={grp} rouge={ROUGE_IMPL}",
        flush=True,
    )

    tap = QueryTap(model, LAYERS, n_q, hd)
    rots = [random_orthogonal(hd, rng) for _ in range(N_ROT_DRAWS)]
    docs = []

    for di, rec in enumerate(data):
        text = build_prompt(tok, rec, prompt_fmt)
        ids = tok(text, truncation=False, return_tensors="pt").to(device)
        T = ids.input_ids.shape[-1]
        if T <= HARNESS.HOT + 64:
            print(f"[doc {di}] too short ({T}), skipped", flush=True)
            continue

        # ---- arm 1: fp16 prefill, greedy 512. y* and NLL_fp16 come together,
        # because y* is by construction fp16's own argmax path.
        tap.on = False
        out = model(**ids, use_cache=True)
        cache_a = out.past_key_values
        k_fp16 = {
            li: cache_keys(cache_a, li)[0][0].float().cpu().numpy() for li in LAYERS
        }
        y_star, nll_fp16 = greedy(
            model, tok, cache_a, out.logits[0, -1], MAXGEN, device
        )
        del cache_a
        torch.cuda.empty_cache()

        # ---- arm 2: nf4a prefill, teacher-forced on y*, queries captured.
        tap.on = True
        out_q = model(**ids, use_cache=True)
        tap.on = False
        cache_b = quantize_prefill(out_q.past_key_values, HARNESS.HOT)
        k_nf4a = {
            li: cache_keys(cache_b, li)[0][0].float().cpu().numpy() for li in LAYERS
        }
        first_lp = torch.log_softmax(out_q.logits[0, -1].float(), -1)
        yt = torch.tensor([y_star], device=device)
        pos = torch.arange(T, T + len(y_star), device=device).unsqueeze(0)
        tap.on = True
        out_tf = model(
            input_ids=yt,
            past_key_values=cache_b,
            position_ids=pos,
            use_cache=True,
        )
        tap.on = False
        q_dec = {
            li: tap.rotated(li, pos, out_tf.logits)[0].float().cpu().numpy()
            for li in LAYERS
        }
        lp = torch.log_softmax(out_tf.logits[0].float(), -1)
        nll_nf4a = [-float(first_lp[y_star[0]])] + [
            -float(lp[i, y_star[i + 1]]) for i in range(len(y_star) - 1)
        ]
        del cache_b, out_tf, out_q
        torch.cuda.empty_cache()

        # ---- arm 3: nf4a prefill, free-running greedy, for A0.
        out3 = model(**ids, use_cache=True)
        cache_c = quantize_prefill(out3.past_key_values, HARNESS.HOT)
        y_nf4a, _ = greedy(model, tok, cache_c, out3.logits[0, -1], MAXGEN, device)
        del cache_c, out3
        torch.cuda.empty_cache()

        refs = rec["answers"]
        r_fp16 = rouge_l(tok.decode(y_star, skip_special_tokens=True), refs)
        r_nf4a = rouge_l(tok.decode(y_nf4a, skip_special_tokens=True), refs)

        # ---- geometry, per (layer, kv-head) cell -----------------------------
        n_settled = T - HARNESS.HOT
        cells = []
        for li in LAYERS:
            Kf, Kq = k_fp16[li], k_nf4a[li]
            Qd = q_dec[li]
            for h in range(n_kv):
                Kfh = Kf[h][:n_settled]
                delta = Kq[h][:n_settled] - Kfh
                Sig = delta.T @ delta / max(1, n_settled)
                probe = rng.choice(
                    n_settled, size=min(PROBE_KEYS, n_settled), replace=False
                )
                qh = Qd[h * grp : (h + 1) * grp]  # (grp, S, hd)
                edges = np.linspace(0, qh.shape[1], N_WINDOWS + 1, dtype=int)
                dw, dw_rot = [], []
                for w in range(N_WINDOWS):
                    Qw = np.ascontiguousarray(
                        qh[:, edges[w] : edges[w + 1], :].reshape(-1, hd)
                    )
                    M = C11C.operator(Kfh, Qw, probe, hd)
                    trM = float(np.trace(M))
                    if trM <= 0:
                        dw.append(0.0)
                        dw_rot.append([0.0] * N_ROT_DRAWS)
                        continue
                    dw.append(float(np.sum(M * Sig)) / trM)
                    dw_rot.append(
                        [float(np.sum(M * (R @ Sig @ R.T))) / trM for R in rots]
                    )
                cells.append(
                    {
                        "layer": li,
                        "kv_head": h,
                        "d_windows": dw,
                        "d_rot_windows": dw_rot,
                        "sigma_trace": float(np.trace(Sig)),
                    }
                )

        n = min(len(nll_fp16), len(nll_nf4a))
        d_tf = [nll_nf4a[i] - nll_fp16[i] for i in range(n)]
        e0, e1 = EARLY
        l0, l1 = LATE
        early_tf = float(np.mean(d_tf[e0:e1])) if n > e1 else float("nan")
        late_tf = float(np.mean(d_tf[l0:l1])) if n > l0 else float("nan")

        def growth(key):
            gs = []
            for c in cells:
                a, b = c["d_windows"][0], c["d_windows"][-1]
                if key == "rot":
                    a = float(np.mean([r for r in c["d_rot_windows"][0]]))
                    b = float(np.mean([r for r in c["d_rot_windows"][-1]]))
                if a > 0:
                    gs.append((b - a) / a)
            return float(np.median(gs)) if gs else float("nan")

        rowd = {
            "doc": di,
            "prompt_tokens": int(T),
            "n_settled": int(n_settled),
            "n_gen_fp16": len(y_star),
            "n_gen_nf4a": len(y_nf4a),
            "rouge_fp16": r_fp16,
            "rouge_nf4a": r_nf4a,
            "rouge_gap": r_fp16 - r_nf4a,
            "d_tf_early": early_tf,
            "d_tf_late": late_tf,
            "d_tf_rise": late_tf - early_tf,
            "d_tf_curve": [
                float(np.mean(d_tf[i : i + 64])) for i in range(0, n - 63, 64)
            ],
            "g_positional": growth("pos"),
            "g_rotated": growth("rot"),
            "cells": cells,
        }
        docs.append(rowd)
        print(
            f"[doc {di}] T={T} gen={len(y_star)}/{len(y_nf4a)} "
            f"rouge {r_fp16:.1f}/{r_nf4a:.1f} gap {r_fp16 - r_nf4a:+.1f}  "
            f"D_tf {early_tf:.4f}->{late_tf:.4f} ({late_tf - early_tf:+.4f})  "
            f"g {rowd['g_positional']:+.3f} vs rot {rowd['g_rotated']:+.3f}",
            flush=True,
        )
        (OUTDIR / "partial.json").write_text(json.dumps(docs))

    tap.close()
    if not docs:
        print("no documents graded")
        return 1

    # ---- bars ---------------------------------------------------------------
    ok = [d for d in docs if np.isfinite(d["d_tf_rise"])]
    rouge_gap = float(np.mean([d["rouge_fp16"] for d in docs])) - float(
        np.mean([d["rouge_nf4a"] for d in docs])
    )
    rises = [d["d_tf_rise"] for d in ok]
    med_rise = float(np.median(rises))
    p_sign = sign_test_p(rises, 0.0)
    g_pos = [d["g_positional"] for d in ok if np.isfinite(d["g_positional"])]
    g_rot = [d["g_rotated"] for d in ok if np.isfinite(d["g_rotated"])]
    b1 = float(np.median(g_pos)) - float(np.median(g_rot))
    rho, p_rho = spearman(
        [d["g_positional"] for d in ok], [d["d_tf_rise"] for d in ok]
    )
    rho_rot, p_rot = spearman(
        [d["g_rotated"] for d in ok], [d["d_tf_rise"] for d in ok]
    )

    bars = {
        "A0_phenomenon_reproduces": bool(rouge_gap >= A0_ROUGE_BAR),
        "A1_survives_teacher_forcing": bool(
            med_rise >= A1_NATS_BAR and p_sign < 0.05
        ),
        "B1_orientation_does_work": bool(b1 >= B1_GROWTH_BAR),
        "B2_geometry_predicts_damage": bool(
            rho > 0 and p_rho < B2_ALPHA and rho > rho_rot
        ),
    }
    if not bars["A0_phenomenon_reproduces"]:
        verdict = "VOID"
    elif all(bars.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    record = {
        "schema": "readscope-c12-longgen-drift-v1",
        "declaration": "calibration/DECLARATION-C12.md",
        "declaration_commit": "90e2ce2",
        "provenance": "tests C-11c's drift mechanism against the degradation "
        "curve in turboquant-pro benchmarks/kvquant_matrix/results_longgen.json; "
        "quantization imported from tq_paper_lb_shard.py and the read operator "
        "imported from c11c_operator_drift.py, so neither is re-derived here",
        "harness_env": HARNESS_ENV,
        "declared": {
            "n_docs": N_DOCS,
            "layers": LAYERS,
            "maxgen": MAXGEN,
            "early_window": list(EARLY),
            "late_window": list(LATE),
            "n_windows": N_WINDOWS,
            "n_rot_draws": N_ROT_DRAWS,
            "probe_keys": PROBE_KEYS,
            "seed": SEED,
            "a0_rouge_bar": A0_ROUGE_BAR,
            "a1_nats_bar": A1_NATS_BAR,
            "b1_growth_bar": B1_GROWTH_BAR,
            "b2_alpha": B2_ALPHA,
            "null": "random rotation of Sigma_delta, identical spectrum, "
            "arbitrary orientation; an isotropic null is constant by "
            "construction and so carries no information",
        },
        "rouge_impl": ROUGE_IMPL,
        "summary": {
            "n_docs_graded": len(docs),
            "rouge_fp16": float(np.mean([d["rouge_fp16"] for d in docs])),
            "rouge_nf4a": float(np.mean([d["rouge_nf4a"] for d in docs])),
            "rouge_gap": rouge_gap,
            "median_d_tf_early": float(np.median([d["d_tf_early"] for d in ok])),
            "median_d_tf_late": float(np.median([d["d_tf_late"] for d in ok])),
            "median_d_tf_rise": med_rise,
            "sign_test_p": p_sign,
            "median_g_positional": float(np.median(g_pos)),
            "median_g_rotated": float(np.median(g_rot)),
            "b1_margin": b1,
            "spearman_g_vs_rise": rho,
            "spearman_p": p_rho,
            "spearman_rot_vs_rise": rho_rot,
            "spearman_rot_p": p_rot,
        },
        "docs": docs,
        "bars": bars,
        "verdict": {"value": verdict, "computed_from": sorted(bars)},
        "runtime": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "code_commit": os.environ.get("CODE_COMMIT", "unknown"),
        },
    }
    out = OUTDIR / "c12-longgen-drift.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"\nROUGE-L fp16 {record['summary']['rouge_fp16']:.2f} "
          f"nf4a {record['summary']['rouge_nf4a']:.2f} "
          f"gap {rouge_gap:+.2f} (bar {A0_ROUGE_BAR})")
    print(f"teacher-forced D_tf {record['summary']['median_d_tf_early']:.4f} -> "
          f"{record['summary']['median_d_tf_late']:.4f} "
          f"rise {med_rise:+.4f} nats, sign p={p_sign:.4f} (bar {A1_NATS_BAR})")
    print(f"geometric growth positional {np.median(g_pos):+.4f} vs rotated "
          f"{np.median(g_rot):+.4f}, margin {b1:+.4f} (bar {B1_GROWTH_BAR})")
    print(f"spearman g vs D_tf rise rho={rho:+.3f} p={p_rho:.4f}; "
          f"null rho={rho_rot:+.3f} p={p_rot:.4f}")
    for k in sorted(bars):
        print(f"{k:<32} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
