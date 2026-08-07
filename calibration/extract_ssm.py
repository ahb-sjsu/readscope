#!/usr/bin/env python3
"""Extract selective-SSM parameters from a real Mamba, for C-3c.

Not a calibration. The data step, kept separate so the sweep stays numpy-only.

A Mamba mixer runs, per channel ``c`` and state dimension ``n``,

    h_t[n] = exp(A[c, n] * dt[c, t]) h_{t-1}[n] + dB_t[c, n] x_t[c]
    y_t[c] = sum_n C_t[n] h_t[c, n] + D[c] x_t[c]

so the objects that decide what a readout sees are ``A``, the per-channel
per-step ``dt``, and the per-step readout vectors ``C_t``. ``B`` and ``C`` are
shared across channels in this architecture and only ``dt`` is per-channel,
which is worth knowing before reading any result.

They are captured from a real forward pass: a hook on ``x_proj`` gives the
raw time-step, ``B`` and ``C`` splits, ``dt_proj`` plus softplus gives the
discretisation step, and ``A_log`` comes from the weights.

Runs on Atlas, CPU only.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

OUT = Path("/archive/readscope/ssm")
MODEL = "state-spaces/mamba-790m-hf"
TAG = "mamba-790m"
LAYER_FRACTIONS = [0.25, 0.5, 0.75]
CHANNELS = [0, 1, 2, 3]
SEQ = 192

TEXT = (
    "The instrument is trusted because it is specified. A recurrence "
    "carries its past forward with a decay, so what it can still see of "
    "an old input depends on how much of that input has already leaked "
    "away. Engineers call the same effect memory length, and they put it "
    "on the datasheet rather than discovering it in the field. "
) * 8


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    backbone = model.backbone
    n_layers = len(backbone.layers)
    layers = [
        max(0, min(n_layers - 1, int(round(f * n_layers))))
        for f in LAYER_FRACTIONS
    ]

    captured: dict[int, dict] = {}
    handles = []

    def make_hook(idx, store=captured):
        def hook(_mod, _inp, out):
            store.setdefault(idx, {})["x_proj"] = out.detach()

        return hook

    for li in layers:
        handles.append(
            backbone.layers[li].mixer.x_proj.register_forward_hook(
                make_hook(li)
            )
        )

    ids = tok(TEXT, return_tensors="pt").input_ids[:, :SEQ]
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    manifest = []
    for frac, li in zip(LAYER_FRACTIONS, layers, strict=False):
        mixer = backbone.layers[li].mixer
        proj = captured[li]["x_proj"]  # (B, L, dt_rank + 2 * d_state)
        if proj.dim() == 3 and proj.shape[1] != ids.shape[1]:
            proj = proj.transpose(1, 2)
        dt_rank = mixer.time_step_rank
        d_state = mixer.ssm_state_size
        raw_dt, B_t, C_t = torch.split(
            proj[0], [dt_rank, d_state, d_state], dim=-1
        )
        with torch.no_grad():
            dt = torch.nn.functional.softplus(
                mixer.dt_proj(raw_dt) + mixer.dt_proj.bias * 0.0
            )  # (L, d_inner)
            A = -torch.exp(mixer.A_log.float())  # (d_inner, d_state)

        A_np = A.cpu().numpy().astype(np.float64)
        dt_np = dt.cpu().numpy().astype(np.float64)
        C_np = C_t.cpu().numpy().astype(np.float64)

        for ch in CHANNELS:
            if ch >= A_np.shape[0]:
                continue
            name = f"{TAG}_L{li}_C{ch}.npz"
            np.savez_compressed(
                OUT / name,
                A=A_np[ch],  # (d_state,)
                dt=dt_np[:, ch],  # (L,)
                C=C_np,  # (L, d_state)
            )
            manifest.append(
                {
                    "file": name,
                    "model": MODEL,
                    "tag": TAG,
                    "family": "mamba",
                    "layer": li,
                    "layer_fraction": frac,
                    "channel": ch,
                    "n_layers": int(n_layers),
                    "seq": int(dt_np.shape[0]),
                    "d_state": int(d_state),
                    "note": "B and C are shared across channels; only dt is "
                    "per-channel",
                }
            )
        print(
            f"  L{li} d_state {d_state} seq {dt_np.shape[0]} "
            f"channels {CHANNELS}",
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
