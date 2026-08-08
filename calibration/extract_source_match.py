#!/usr/bin/env python3
"""Extract Llama-3.2-3B activations matched to the source program's protocol.

Not a calibration. The data step for C-10, which closes C-4's unmatched
remainder.

C-4 found that the published 0.647 measures agreement between two references
rather than probe fidelity, and that the reference choice explains much but
not all of the gap. The unmatched factors were the query capture, the
grouped-query grouping and the model and layer set. This matches all three to
`gateB_llama_rematch.py`.

The one that matters most is the query set. That script uses

    Qset = q[b, h * grp : (h + 1) * grp].reshape(-1, d)

so the query set for a key-value head is **every query in its group across
the whole sequence**, ``grp * S`` vectors, not a sample. With three query
heads per key-value head and 192 positions that is 576 queries spanning the
full 128-dimensional head space, where C-4 used 24. A covariance built from
24 vectors has rank 24 and a well-separated top-16; one built from 576
vectors is full rank and its top-16 is far less determined. That difference
is a candidate for the entire unmatched residual.

Layers {8, 16} and float32 are also matched.

Runs on Atlas, CPU only.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_activations_v2 import (  # noqa: E402
    _cache_keys,
    _decoder,
    _rotate_half,
)

OUT = Path("/archive/readscope/source_match")
MODEL = "unsloth/Llama-3.2-3B"
TAG = "llama32-3b"
LAYERS = [8, 16]
SEQ = 192
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

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    dec = _decoder(model)
    captured: dict[int, object] = {}
    handles = []

    def make_hook(idx, store=captured):
        def hook(_mod, _inp, out):
            store[idx] = out.detach()

        return hook

    for li in LAYERS:
        handles.append(
            dec.layers[li].self_attn.q_proj.register_forward_hook(
                make_hook(li)
            )
        )

    ids = tok(TEXT, return_tensors="pt").input_ids[:, :SEQ]
    with torch.no_grad():
        fwd = model(ids, use_cache=True)
    for h in handles:
        h.remove()
    past = fwd.past_key_values
    S = ids.shape[1]
    pos = torch.arange(S).unsqueeze(0)

    manifest = []
    for li in LAYERS:
        keys = _cache_keys(past, li)[0].float()  # (n_kv, S, d)
        n_kv, _, d = keys.shape

        q_flat = captured[li][0].float()
        n_heads = q_flat.shape[-1] // d
        q = q_flat.view(S, n_heads, d).transpose(0, 1)  # (H, S, d)
        cos, sin = dec.rotary_emb(q.unsqueeze(0), pos)
        q = (q * cos[0].unsqueeze(0)) + (_rotate_half(q) * sin[0].unsqueeze(0))
        grp = max(1, n_heads // n_kv)

        for h in range(n_kv):
            K = np.ascontiguousarray(keys[h].cpu().numpy().astype(np.float32))
            # every query in this key-value head's group, as the source does
            Qset = np.ascontiguousarray(
                q[h * grp : (h + 1) * grp]
                .reshape(-1, d)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            name = f"{TAG}_L{li}_KV{h}.npz"
            np.savez_compressed(OUT / name, K=K, Q=Qset)
            manifest.append(
                {
                    "file": name,
                    "model": MODEL,
                    "tag": TAG,
                    "layer": li,
                    "kv_head": h,
                    "group_size": int(grp),
                    "n_heads": int(n_heads),
                    "n_kv_heads": int(n_kv),
                    "seq": int(S),
                    "head_dim": int(d),
                    "n_queries": int(Qset.shape[0]),
                    "dtype": "float32",
                    "protocol": "matched to gateB_llama_rematch.py",
                }
            )
        print(
            f"  L{li} n_kv {n_kv} grp {grp} d {d} queries/cell " f"{grp * S}",
            flush=True,
        )

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    del model
    gc.collect()
    print("cells:", len(manifest))
    print(OUT / "manifest.json")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
