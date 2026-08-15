#!/usr/bin/env python3
"""C-12 Stage D — the intervention, gated on A0 AND A1 AND B1.

Spec sealed in geometric-observation/crucible/PREREG-OT4-APPENDIX.md:
per (layer, kv-head), rotate the settled prefill keys into the read
operator's eigenbasis, allocate a fixed total bit budget across
directions by reverse water-filling against the operator spectrum,
quantize per direction, rotate back. Arm E allocates against the
early-window operator (decode positions (0,128]), arm U against the
union operator (all decode positions); identical budgets by
construction; values stay fp16 in every Stage D arm so the difference
is keys alone.

D1 (sealed): median over documents of late-window (384,512] teacher-
forced excess NLL falls >= 10% for arm U relative to arm E.

Runs after c12_longgen_drift.py, same env, same OUTDIR:

    CUDA_VISIBLE_DEVICES=1 CODEBOOK=nf4 OUTDIR=/archive/c12/out-sym \
        HARNESS_DIR=$HOME/turboquant-pro/benchmarks/kvquant_matrix \
        $HOME/env/bin/python c12_stage_d.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import c12_longgen_drift as C12  # noqa: E402  (module setup, no run)
import torch  # noqa: E402

OUTDIR = C12.OUTDIR
GATE_RECORD = OUTDIR / "c12-longgen-drift.json"
BITS_PER_DIM = 4.0            # matched to the arms' KEY_BITS
BIT_CAP = 12
EARLY_WINDOW = 128
LATE = C12.LATE
SEED = C12.SEED


def waterfill_bits(lam, budget, cap=BIT_CAP):
    """Integer per-direction bits from reverse water-filling, sum exact."""
    lam = np.clip(np.asarray(lam, float), 1e-300, None)
    lo, hi = 1e-12 * lam.max(), max(lam.max(), 1e-12)
    for _ in range(200):
        th = np.sqrt(lo * hi)
        if np.clip(np.log2(lam / th), 0, cap).sum() < budget:
            hi = th
        else:
            lo = th
    b = np.clip(np.log2(lam / np.sqrt(lo * hi)), 0, cap)
    ib = np.floor(b).astype(int)
    rem = int(round(budget)) - int(ib.sum())
    if rem > 0:
        order = np.argsort(-(b - ib))
        for i in order[:rem]:
            if ib[i] < cap:
                ib[i] += 1
    return ib


def quantize_against(K, M, bits_per_dim=BITS_PER_DIM):
    """Quantize keys K (S, d) with bits allocated by operator M's spectrum."""
    d = K.shape[1]
    lam, V = np.linalg.eigh(M)
    lam, V = lam[::-1], V[:, ::-1]
    bits = waterfill_bits(lam, bits_per_dim * d)
    Y = K @ V
    Yq = np.empty_like(Y)
    for i in range(d):
        col = Y[:, i]
        if bits[i] < 1:
            Yq[:, i] = col.mean()
            continue
        lo, hi = float(col.min()), float(col.max())
        if hi <= lo:
            Yq[:, i] = col
            continue
        levels = (1 << int(bits[i])) - 1
        step = (hi - lo) / levels
        Yq[:, i] = np.round((col - lo) / step) * step + lo
    return (Yq @ V.T).astype(K.dtype), bits


@torch.no_grad()
def main() -> int:
    rec = json.loads(GATE_RECORD.read_text())
    bars = rec["bars"]
    gates = [
        "A0_phenomenon_reproduces",
        "A1_survives_teacher_forcing",
        "B1_orientation_does_work",
    ]
    if not all(bars.get(g) for g in gates):
        print("GATES NOT MET:", {g: bars.get(g) for g in gates})
        return 2

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    prompt_fmt = json.load(open(f"{C12.LB_CONF}/dataset2prompt.json"))[
        "gov_report"
    ]
    data = [json.loads(x) for x in open(C12.LB_DATA)][: C12.N_DOCS]

    tok = C12.AutoTokenizer.from_pretrained(
        C12.HARNESS_ENV["MODEL"], use_fast=True
    )
    model = (
        C12.AutoModelForCausalLM.from_pretrained(
            C12.HARNESS_ENV["MODEL"],
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    grp = n_q // n_kv
    layers_all = list(range(cfg.num_hidden_layers))
    tap = C12.ProjTap(model, layers_all, n_q, n_kv, hd)
    print(f"[stage-d] layers={len(layers_all)} kv={n_kv} hd={hd}", flush=True)

    docs = []
    for di, drec in enumerate(data):
        text = C12.build_prompt(tok, drec, prompt_fmt)
        ids = tok(text, truncation=False, return_tensors="pt").to(device)
        T = ids.input_ids.shape[-1]
        if T <= C12.HARNESS.HOT + 64:
            print(f"[doc {di}] too short ({T}), skipped", flush=True)
            continue
        n_settled = T - C12.HARNESS.HOT
        rng = np.random.default_rng(SEED + 1000 + di)
        t0 = time.time()

        with C12.nf4a_cache(False), C12.EFFICIENT():
            y_star = C12.generate_tokens(model, tok, ids, C12.MAXGEN)
        yt = torch.tensor([y_star], device=device)
        pos = torch.arange(T, T + len(y_star), device=device).unsqueeze(0)

        def tf_pass(key_patch=None, capture=False):
            """Teacher-forced NLL on y*; optionally patch settled keys."""
            with C12.nf4a_cache(False), C12.EFFICIENT():
                first, cache = C12.prefill(model, ids.input_ids, device)
                keys = {}
                for li in layers_all:
                    kt = C12.cache_keys(cache, li)[0]
                    keys[li] = kt[0, :, :T, :].float().cpu().numpy()
                    if key_patch is not None:
                        for h in range(n_kv):
                            kt[0, h, :n_settled, :] = torch.from_numpy(
                                key_patch[li][h]
                            ).to(kt.dtype).to(kt.device)
                tap.on = capture
                out = model(
                    input_ids=yt,
                    past_key_values=cache,
                    position_ids=pos,
                    use_cache=True,
                )
                tap.on = False
            flp = torch.log_softmax(first.float(), -1)
            lp = torch.log_softmax(out.logits[0].float(), -1)
            nll = [-float(flp[y_star[0]])] + [
                -float(lp[i, y_star[i + 1]]) for i in range(len(y_star) - 1)
            ]
            qd = None
            if capture:
                qd = {
                    li: tap.rotated_q(li, pos, out.logits)[0]
                    .float()
                    .cpu()
                    .numpy()
                    for li in layers_all
                }
            del cache, out
            torch.cuda.empty_cache()
            return np.array(nll), keys, qd

        nll_fp16, k_fp16, q_dec = tf_pass(capture=True)

        # ---- build both allocations per (layer, kv-head) ----
        patches = {"early": {}, "union": {}}
        bits_used = {"early": [], "union": []}
        for li in layers_all:
            patches["early"][li] = np.empty((n_kv, n_settled, hd), np.float32)
            patches["union"][li] = np.empty((n_kv, n_settled, hd), np.float32)
            for h in range(n_kv):
                Kfh = k_fp16[li][h][:n_settled]
                qh = q_dec[li][h * grp : (h + 1) * grp]
                probe = rng.choice(
                    n_settled,
                    size=min(C12.PROBE_KEYS, n_settled),
                    replace=False,
                )
                s_len = qh.shape[1]
                for arm, sl in (
                    ("early", slice(0, min(EARLY_WINDOW, s_len))),
                    ("union", slice(0, s_len)),
                ):
                    Qw = np.ascontiguousarray(
                        qh[:, sl, :].reshape(-1, hd)
                    )
                    M = C12.operator_fast(Kfh, Qw, probe, hd)
                    Khat, bits = quantize_against(Kfh, M)
                    patches[arm][li][h] = Khat
                    bits_used[arm].append(int(bits.sum()))

        nll_e, _, _ = tf_pass(key_patch=patches["early"])
        nll_u, _, _ = tf_pass(key_patch=patches["union"])

        def late_excess(nll_arm):
            d_tf = nll_arm - nll_fp16
            lo, hi = LATE
            hi = min(hi, len(d_tf))
            return float(np.mean(d_tf[lo:hi])) if hi > lo else float("nan")

        row = {
            "doc": di,
            "T": T,
            "n_gen": len(y_star),
            "late_excess_early_alloc": late_excess(nll_e),
            "late_excess_union_alloc": late_excess(nll_u),
            "bits_match": bits_used["early"] == bits_used["union"],
            "timing_s": round(time.time() - t0, 1),
        }
        docs.append(row)
        print(
            f"[doc {di}] T={T} late D_tf: early-alloc "
            f"{row['late_excess_early_alloc']:+.4f}  union-alloc "
            f"{row['late_excess_union_alloc']:+.4f}  "
            f"bits_match={row['bits_match']}  ({row['timing_s']}s)",
            flush=True,
        )

    ok = [
        d
        for d in docs
        if np.isfinite(d["late_excess_early_alloc"])
        and np.isfinite(d["late_excess_union_alloc"])
    ]
    med_e = float(np.median([d["late_excess_early_alloc"] for d in ok]))
    med_u = float(np.median([d["late_excess_union_alloc"] for d in ok]))
    drop = (med_e - med_u) / med_e if med_e > 0 else float("nan")
    d1 = bool(med_e > 0 and drop >= 0.10)
    bits_all_match = all(d["bits_match"] for d in docs)

    record = {
        "schema": "readscope-c12-stage-d-v1",
        "gates": {g: bool(bars.get(g)) for g in gates},
        "docs": docs,
        "summary": {
            "n_docs": len(ok),
            "median_late_excess_early_alloc": med_e,
            "median_late_excess_union_alloc": med_u,
            "relative_drop": drop,
            "bits_matched_everywhere": bits_all_match,
        },
        "bars": {"D1_union_cuts_late_damage": d1},
    }
    (OUTDIR / "c12-stage-d.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)
    )
    print(
        f"\nD1: late D_tf median early-alloc {med_e:.4f} -> union-alloc "
        f"{med_u:.4f}  drop {drop:+.1%} (bar 10%)  "
        f"bits matched everywhere: {bits_all_match}"
    )
    print("D1", "PASS" if d1 else "FAIL")
    return 0 if d1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
