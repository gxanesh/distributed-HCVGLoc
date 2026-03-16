#!/bin/bash
# run_single_gpu.sh — Single GPU baseline training (for comparison)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-configs/stage1_baseline.yaml}"

echo "============================================================"
echo "  HCVGLoc Stage 1 — Single GPU Baseline"
echo "  Config : $CONFIG"
echo "============================================================"

cd "$PROJECT_ROOT"
python tools/train_stage1.py --config "$CONFIG"
echo "Done."
