#!/usr/bin/env python3
"""Extract real post-RoPE keys AND real post-RoPE queries.

Supersedes extract_activations.py, whose query set was drawn from the key
stream. That simplification saturated the softmax on the self-match term and
drove median attention entropy to 0.152 bits across 42 cells, which broke
three of C-3's bars for a reason that had nothing to do with the models. The
artifact and its diagnosis stay in the record.

Here the queries are the model's own. A forward hook on ``q_proj`` captures
the pre-rotary query projection, and the model's own ``rotary_emb`` module
supplies the cosines and sines, so what comes out is the query tensor the
attention softmax actually consumed. Grouped-query attention is handled by
mapping each attention head to the key-value head it reads.

Runs on Atlas, CPU only, threads capped for thermal headroom.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("/archive/readscope/activations_v2")

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


def _text_config(config):
    for holder in (config, getattr(config, "text_config", None)):
        if holder is not None and hasattr(holder, "num_hidden_layers"):
            return holder
    raise AttributeError("no num_hidden_layers on config or text_config")


def _cache_keys(past, layer):
    if hasattr(past, "layers"):
        return past.layers[layer].keys
    if hasattr(past, "key_cache"):
        return past.key_cache[layer]
    return past[layer][0]


def _decoder(model):
    """The module that owns .layers and .rotary_emb."""
    for path in ("model.language_model", "model", "model.model"):
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "layers"):
            return obj
    raise AttributeError("could not locate the decoder stack")


def _rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for spec in MODELS:
        if ONLY and spec["tag"] != ONLY:
            continue
        print("=== " + spec["id"], flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(spec["id"])
            model = AutoModelForCausalLM.from_pretrained(
                spec["id"], dtype=torch.float32
            )
        except Exception as exc:  # noqa: BLE001
            print("  SKIP, load failed:", repr(exc)[:180], flush=True)
            continue

        model.eval()
        handles = []
        captured = {}
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
                keys = _cache_keys(past, li)[0].float()  # (n_kv, S, d)
                n_kv, _, d = keys.shape

                q_flat = captured[li][0].float()  # (S, n_heads * d)
                n_heads = q_flat.shape[-1] // d
                q = q_flat.view(S, n_heads, d).transpose(0, 1)  # (H, S, d)

                # the model's own rotary embedding, applied as the attention
                # module applies it
                # Gemma-3 keys its rotary tables by layer type (sliding vs
                # full attention) and errors on a bare call; every other
                # family here takes the two-argument form.
                # Gemma-3 keeps separate rotary tables for sliding and
                # full attention and looks them up by the layer's type,
                # which lives on the config rather than on the layer.
                types = getattr(cfg, "layer_types", None)
                layer_type = types[li] if types else None
                if layer_type is None:
                    cos, sin = dec.rotary_emb(q.unsqueeze(0), pos)
                else:
                    cos, sin = dec.rotary_emb(
                        q.unsqueeze(0), pos, layer_type=layer_type
                    )
                cos = cos[0].unsqueeze(0)  # (1, S, d)
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
                            "family": spec["family"],
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
                            "queries": "real, q_proj hook + model rotary_emb",
                            "layer_type": layer_type,
                        }
                    )
            print(
                f"  saved layers {layers}, head_dim {d}, "
                f"heads {n_heads}, kv {n_kv}",
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
