#!/usr/bin/env python3
"""Extract keys and queries across the Qwen2.5 scale ladder, for C-5.

Not a calibration. The data step.

Qwen2.5 keeps ``head_dim`` at 128 from 1.5B through 32B while layer count,
head count and grouping all change, so this ladder varies **scale with the
geometry held fixed**. Whatever moves is about the substrate and not about
the dimension the probe works in, which is the only way to ask the scale
question cleanly.

**Declared: every model is loaded in bfloat16.** Loading the small ones in
float32 and the large ones in bfloat16 would confound precision with scale,
and bfloat16 is what these models are served in. Activations are cast to
float64 before any analysis, and the analytic ground truth is computed from
the same activations, so the grading stays exact; what bfloat16 changes is
the substrate, not the comparison.

Reuses the query capture from ``extract_activations_v2``: a hook on
``q_proj`` plus the model's own rotary embedding, with grouped-query
attention resolved by mapping each attention head to the key-value head it
reads.

Runs on Atlas, CPU only, threads capped for thermal headroom.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_activations_v2 import (  # noqa: E402
    _cache_keys,
    _decoder,
    _rotate_half,
    _text_config,
)

OUT = Path("/archive/readscope/ladder")
ONLY = os.environ.get("RS_ONLY", "").strip()

MODELS = [
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "tag": "qwen25-1.5b",
        "params_b": 1.5,
    },
    {"id": "Qwen/Qwen2.5-7B-Instruct", "tag": "qwen25-7b", "params_b": 7.0},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "tag": "qwen25-14b", "params_b": 14.0},
    {"id": "Qwen/Qwen2.5-32B-Instruct", "tag": "qwen25-32b", "params_b": 32.0},
]

LAYER_FRACTIONS = [0.25, 0.5, 0.75]
HEADS = [0, 1, 2, 3]
SEQ = 192
N_QUERIES = 24
THREADS = 4

TEXT = (
    "The instrument is trusted because it is specified. A measurement "
    "without its noise floor is a number without a meaning, and a "
    "calibration curve is what turns a demonstration into a device. "
    "Consider a probe attached to a circuit: the act of measuring draws "
    "current, so what the operator reads is never quite what was there "
    "before the probe arrived. Engineers do not treat this as a scandal. "
    "They characterise it, publish it on the datasheet, and correct for "
    "it. The same discipline applies to any instrument that couples to "
    "the system it observes, whether that system is an amplifier stage "
    "or the attention mechanism of a language model reading its own "
    "cache of keys and queries across many layers and many heads. "
) * 6


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(THREADS)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for spec in MODELS:
        if ONLY and spec["tag"] != ONLY:
            continue
        print("=== " + spec["id"], flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(spec["id"])
            model = AutoModelForCausalLM.from_pretrained(
                spec["id"], dtype=torch.bfloat16
            )
        except Exception as exc:  # noqa: BLE001
            print("  SKIP, load failed:", repr(exc)[:180], flush=True)
            continue

        model.eval()
        handles = []
        captured: dict[int, object] = {}
        try:
            cfg = _text_config(model.config)
            n_layers = int(cfg.num_hidden_layers)
            dec = _decoder(model)
            layers = [
                max(0, min(n_layers - 1, int(round(f * n_layers))))
                for f in LAYER_FRACTIONS
            ]

            def make_hook(idx, store=captured):
                def hook(_mod, _inp, out):
                    store[idx] = out.detach()

                return hook

            for li in layers:
                handles.append(
                    dec.layers[li].self_attn.q_proj.register_forward_hook(
                        make_hook(li)
                    )
                )

            ids = tok(TEXT, return_tensors="pt").input_ids[:, :SEQ]
            with torch.no_grad():
                fwd = model(ids, use_cache=True)
            past = fwd.past_key_values
            S = ids.shape[1]
            pos = torch.arange(S).unsqueeze(0)

            for frac, li in zip(LAYER_FRACTIONS, layers, strict=False):
                keys = _cache_keys(past, li)[0].float()
                n_kv, _, d = keys.shape

                q_flat = captured[li][0].float()
                n_heads = q_flat.shape[-1] // d
                q = q_flat.view(S, n_heads, d).transpose(0, 1)

                cos, sin = dec.rotary_emb(q.unsqueeze(0), pos)
                cos = cos[0].unsqueeze(0)
                sin = sin[0].unsqueeze(0)
                q = (q * cos) + (_rotate_half(q) * sin)

                group = max(1, n_heads // n_kv)
                idx = np.linspace(0, S - 1, N_QUERIES, dtype=int)

                for head in HEADS:
                    if head >= n_heads:
                        continue
                    kv_head = head // group
                    K = np.ascontiguousarray(
                        keys[kv_head].cpu().numpy().astype(np.float64)
                    )
                    Q = np.ascontiguousarray(
                        q[head].cpu().numpy().astype(np.float64)[idx]
                    )
                    name = f"{spec['tag']}_L{li}_H{head}.npz"
                    np.savez_compressed(OUT / name, K=K, Q=Q)
                    manifest.append(
                        {
                            "file": name,
                            "model": spec["id"],
                            "tag": spec["tag"],
                            "family": "qwen",
                            "params_b": spec["params_b"],
                            "layer": li,
                            "layer_fraction": frac,
                            "head": head,
                            "kv_head": int(kv_head),
                            "n_heads": int(n_heads),
                            "n_kv_heads": int(n_kv),
                            "n_layers": n_layers,
                            "seq": int(S),
                            "head_dim": int(d),
                            "n_queries": int(N_QUERIES),
                            "dtype": "bfloat16",
                        }
                    )
            print(
                f"  saved layers {layers}, head_dim {d}, heads {n_heads}, "
                f"kv {n_kv}, n_layers {n_layers}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print("  SKIP, extraction failed:", repr(exc)[:220], flush=True)
        finally:
            for h in handles:
                h.remove()
            del model
            captured.clear()
            gc.collect()

    if ONLY and (OUT / "manifest.json").exists():
        prior = json.loads((OUT / "manifest.json").read_text())
        manifest = [c for c in prior if c["tag"] != ONLY] + manifest
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("cells:", len(manifest))
    print(OUT / "manifest.json")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
