#!/bin/bash
# run_ddp_4gpu.sh — Launch 4-GPU DDP training on Ada 6000 workstation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-configs/stage1_ddp_4gpu.yaml}"
RESUME="${2:-}"

echo "============================================================"
echo "  HCVGLoc Stage 1 — 4-GPU DDP Training"
echo "  Config  : $CONFIG"
echo "  Resume  : ${RESUME:-none}"
echo "============================================================"

cd "$PROJECT_ROOT"

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    tools/train_stage1.py \
    --config "$CONFIG" \
    $RESUME_ARG

echo "Done."
