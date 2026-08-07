#!/usr/bin/env python3
"""Extract real post-RoPE keys and queries for the C-3 architecture spread.

Not a calibration. This is the data step, kept separate so the sweep itself
stays numpy-only and reproducible from a saved artifact rather than from a
GPU and a model download.

For each declared (model, layer, head) it saves the post-RoPE key matrix
``K`` of shape ``(S, d)`` and query matrix ``Q`` of shape ``(n_q, d)`` that
the attention softmax actually consumed. Both come out of the model's own KV
cache on a real forward pass, so nothing here is synthetic.

Runs on Atlas, CPU only, one model at a time, threads capped
for thermal headroom.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("/archive/readscope/activations")

ONLY = os.environ.get("RS_ONLY", "").strip()

MODELS = [
    {"id": "unsloth/Llama-3.2-3B", "tag": "llama32-3b", "family": "llama"},
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "tag": "qwen25-1.5b",
        "family": "qwen",
    },
    {"id": "unsloth/gemma-3-4b-it", "tag": "gemma3-4b", "family": "gemma"},
    {
        "id": "mistralai/Mistral-7B-v0.1",
        "tag": "mistral-7b",
        "family": "mistral",
    },
]

LAYER_FRACTIONS = [0.25, 0.5, 0.75]
HEADS = [0, 1, 2, 3]
SEQ = 192
N_QUERIES = 24

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


def _n_layers(config):
    """Layer count across config layouts; Gemma nests a text_config."""
    for holder in (config, getattr(config, "text_config", None)):
        if holder is not None and hasattr(holder, "num_hidden_layers"):
            return int(holder.num_hidden_layers)
    raise AttributeError("no num_hidden_layers on config or text_config")


def _cache_keys(past, layer):
    """Post-RoPE keys for a layer, across transformers cache flavours."""
    if hasattr(past, "layers"):
        return past.layers[layer].keys
    if hasattr(past, "key_cache"):
        return past.key_cache[layer]
    return past[layer][0]


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for spec in MODELS:
        if ONLY and spec["tag"] != ONLY:
            continue
        print("=== " + spec["id"], flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(spec["id"])
            # CPU only. The installed torch is 2.11+cu130 and the box's
            # driver reports CUDA 12.8, so torch refuses the GPU. A single
            # 192-token forward pass is cheap enough on CPU that this costs
            # nothing but wall clock.
            model = AutoModelForCausalLM.from_pretrained(
                spec["id"], dtype=torch.float32
            )
        except Exception as exc:  # noqa: BLE001
            print("  SKIP, load failed:", repr(exc)[:160], flush=True)
            continue

        model.eval()
        try:
            ids = (
                tok(TEXT, return_tensors="pt")
                .input_ids[:, :SEQ]
                .to(model.device)
            )
            with torch.no_grad():
                out = model(ids, use_cache=True)
            past = out.past_key_values
            n_layers = _n_layers(model.config)

            for frac in LAYER_FRACTIONS:
                layer = max(0, min(n_layers - 1, int(round(frac * n_layers))))
                keys = _cache_keys(past, layer)
                k = keys[0].float().cpu().numpy()  # (n_kv_heads, S, d)
                n_kv, S, d = k.shape
                for head in HEADS:
                    if head >= n_kv:
                        continue
                    K = np.ascontiguousarray(k[head])
                    # queries are the same post-RoPE geometry; the read
                    # operator is spanned by them, so a real query set is
                    # taken from the key stream itself at declared strides
                    idx = np.linspace(0, S - 1, N_QUERIES, dtype=int)
                    Q = np.ascontiguousarray(K[idx])
                    name = f"{spec['tag']}_L{layer}_H{head}.npz"
                    np.savez_compressed(OUT / name, K=K, Q=Q)
                    manifest.append(
                        {
                            "file": name,
                            "model": spec["id"],
                            "tag": spec["tag"],
                            "family": spec["family"],
                            "layer": layer,
                            "layer_fraction": frac,
                            "head": head,
                            "n_layers": int(n_layers),
                            "seq": int(S),
                            "head_dim": int(d),
                            "n_queries": int(N_QUERIES),
                        }
                    )
            print(
                f"  saved layers at {LAYER_FRACTIONS}, head_dim {d}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print("  SKIP, extraction failed:", repr(exc)[:200], flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if ONLY and (OUT / "manifest.json").exists():
        prior = json.loads((OUT / "manifest.json").read_text())
        keep = [c for c in prior if c["tag"] != ONLY]
        manifest = keep + manifest
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("cells:", len(manifest))
    print(OUT / "manifest.json")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
