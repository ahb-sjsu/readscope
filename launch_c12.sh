#!/bin/bash
# C-12: drift against the real long-generation degradation curve.
# Bars sealed in calibration/DECLARATION-C12.md at 90e2ce2.
#
#   screen -dmS c12 -L -Logfile /archive/c12/c12.log bash ~/readscope-c12/launch_c12.sh
#
# GPU 1 only: GPU 0 carries Erebus. Single-process inference, so no thermal
# controller is needed, but the BLAS threads are capped anyway.
set -u
cd "$HOME/readscope-c12" || exit 1

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HARNESS_DIR=/home/claude/tqp-c12/benchmarks/kvquant_matrix
export OUTDIR=${OUTDIR:-/archive/c12/out}
export N_DOCS=${N_DOCS:-40}
export CODE_COMMIT=$(git rev-parse --short HEAD)

echo "=== C-12 start $(date -u) commit $CODE_COMMIT n_docs=$N_DOCS ==="
/archive/kvbench/venv/bin/python calibration/c12_longgen_drift.py
echo "=== C-12 exit $? $(date -u) ==="
