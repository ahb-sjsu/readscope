#!/usr/bin/env python3
"""Right-size the activation-extraction job before submitting it to NRP.

Not a calibration. This is the pre-flight that decides whether the job has a
legal resource request at all, and what it is.

NRP judges a pod against what it *requested*, averaged over roughly five
minutes, and the bands are hard: CPU 20 to 200 percent, memory 20 to 150
percent, GPU above 40 percent. Requests equal limits, so the request must be
at least the peak or the pod OOMs, and at most the trough over 0.20 or the
floor is violated. **A workload whose peak exceeds five times its trough has
no legal request.** That is the crossing the resident right-sizer reports as
IMPOSSIBLE, and it is exactly the shape a job has when it spends minutes
downloading a model at near-zero memory and then jumps to tens of gigabytes.

So this measures three things per model.

  threads   the thread count where wall time stops improving, from
            ``batch_probe.probe_threads`` for the thermal ceiling on this box
            and a wall-time sweep for the efficiency knee. The CPU request is
            set from it so utilisation sits near 100 percent of request
            rather than in the violation band.
  peak      maximum resident set size during load and forward.
  trough    minimum resident set size after the process is up, which is what
            the memory floor is measured against.

It then prints the legal window, or says IMPOSSIBLE and why.

Runs on Atlas. Read-only with respect to the cluster; submits nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path("/archive/readscope/rightsize.json")

MODELS = [
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "tag": "qwen25-1.5b"},
    {"id": "unsloth/gemma-3-4b-it", "tag": "gemma3-4b"},
    {"id": "state-spaces/mamba-790m-hf", "tag": "mamba-790m"},
]

THREAD_SWEEP = [2, 4, 8]
SEQ = 192

# NRP bands, from the cluster policy
CPU_FLOOR, CPU_CEIL = 0.20, 2.00
MEM_FLOOR, MEM_CEIL = 0.20, 1.50


def sample_rss(stop, out, pid):
    """Poll this process's RSS in MiB until told to stop."""
    path = f"/proc/{pid}/status"
    while not stop.is_set():
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        out.append(int(line.split()[1]) / 1024.0)
                        break
        except OSError:
            pass
        time.sleep(0.25)


def load_and_forward(model_id, threads):
    """One load plus one forward, subprocessed so RSS stays clean."""
    code = (
        "import torch, os;"
        f"torch.set_num_threads({threads});"
        "from transformers import AutoModelForCausalLM, AutoTokenizer;"
        f"tok=AutoTokenizer.from_pretrained({model_id!r});"
        f"m=AutoModelForCausalLM.from_pretrained({model_id!r},dtype=torch.float32);"
        "m.eval();"
        f"ids=tok('hello world '*80,return_tensors='pt').input_ids[:,:{SEQ}];"
        "import torch as t;\n"
        "with t.no_grad(): m(ids, use_cache=True)\n"
        "print('OK')"
    )
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    samples: list[float] = []
    stop = threading.Event()
    th = threading.Thread(
        target=sample_rss, args=(stop, samples, proc.pid), daemon=True
    )
    th.start()
    out, _ = proc.communicate()
    stop.set()
    th.join(timeout=2)
    return {
        "ok": "OK" in (out or ""),
        "wall_s": time.time() - t0,
        "peak_mib": max(samples) if samples else 0.0,
        "trough_mib": min([s for s in samples if s > 50] or [0.0]),
        "n_samples": len(samples),
    }


def legal_window(peak_mib, trough_mib):
    """The request band, or None when peak exceeds trough over the floor."""
    lo = peak_mib
    hi = trough_mib / MEM_FLOOR if trough_mib > 0 else 0.0
    return (lo, hi) if hi >= lo else None


def main() -> int:
    report = {
        "models": [],
        "bands": {
            "cpu": [CPU_FLOOR, CPU_CEIL],
            "memory": [MEM_FLOOR, MEM_CEIL],
        },
    }

    try:
        from batch_probe import probe_threads

        def stress(n):
            import numpy as np

            os.environ["OMP_NUM_THREADS"] = str(n)
            a = np.random.default_rng(0).standard_normal((1200, 1200))
            end = time.time() + 6
            while time.time() < end:
                a @ a

        safe = probe_threads(
            stress,
            max_temp=80.0,
            low=1,
            high=8,
            work_time=6.0,
            settle_time=3.0,
            cooldown_time=8.0,
            verbose=False,
        )
        report["thermal_safe_threads_atlas"] = int(safe)
        print("batch-probe thermal ceiling on this box:", safe, "threads")
    except Exception as exc:  # noqa: BLE001
        report["thermal_safe_threads_atlas"] = None
        report["thermal_error"] = repr(exc)[:200]
        print("batch-probe thermal probe unavailable:", repr(exc)[:120])

    for spec in MODELS:
        entry = {"model": spec["id"], "tag": spec["tag"], "sweep": []}
        for threads in THREAD_SWEEP:
            r = load_and_forward(spec["id"], threads)
            r["threads"] = threads
            entry["sweep"].append(r)
            print(
                f"  {spec['tag']:<14} t={threads}  wall {r['wall_s']:6.1f}s  "
                f"peak {r['peak_mib']:8.0f} MiB  trough "
                f"{r['trough_mib']:7.0f} MiB  ok={r['ok']}",
                flush=True,
            )
        ok = [r for r in entry["sweep"] if r["ok"]]
        if not ok:
            entry["verdict"] = "LOAD_FAILED"
            report["models"].append(entry)
            continue

        peak = max(r["peak_mib"] for r in ok)
        trough = min(r["trough_mib"] for r in ok)
        best = min(ok, key=lambda r: r["wall_s"])
        win = legal_window(peak, trough)
        entry.update(
            {
                "peak_mib": peak,
                "trough_mib": trough,
                "peak_over_trough": (peak / trough) if trough else None,
                "knee_threads": best["threads"],
                "knee_wall_s": best["wall_s"],
                "legal_memory_window_mib": list(win) if win else None,
                "verdict": "POSSIBLE" if win else "IMPOSSIBLE",
            }
        )
        if win:
            # sit at the peak; requests equal limits so anything less OOMs
            entry["recommended_memory_gi"] = round(
                (peak * 1.15) / 1024.0 + 0.5
            )
            entry["recommended_cpu"] = best["threads"]
        report["models"].append(entry)
        print(
            f"  -> {spec['tag']:<14} {entry['verdict']}  "
            f"peak/trough {entry['peak_over_trough']:.2f}  "
            f"window {entry['legal_memory_window_mib']}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({m["tag"]: m.get("verdict") for m in report["models"]}))
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
