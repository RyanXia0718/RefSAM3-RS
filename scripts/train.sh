#!/usr/bin/env bash

# Minimal two-stage training entry for RefSegRS and RRSIS-D.

set -euo pipefail

DATASET=""
STAGE=""
GPU="0"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train.sh --dataset refsegrs|rrsisd --stage 1|2 [--gpu N]

Examples:
  bash scripts/train.sh --dataset refsegrs --stage 1 --gpu 0
  bash scripts/train.sh --dataset refsegrs --stage 2 --gpu 0
  bash scripts/train.sh --dataset rrsisd   --stage 1 --gpu 1
  bash scripts/train.sh --dataset rrsisd   --stage 2 --gpu 1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --stage)   STAGE="$2"; shift 2 ;;
        --gpu)     GPU="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 2 ;;
    esac
done

case "${DATASET}:${STAGE}" in
    refsegrs:1) CONFIG="configs/sam3i/sam3i_refsegrs_stage1" ;;
    refsegrs:2) CONFIG="configs/sam3i/sam3i_refsegrs_stage2" ;;
    rrsisd:1)   CONFIG="configs/sam3i/sam3i_rrsisd_stage1" ;;
    rrsisd:2)   CONFIG="configs/sam3i/sam3i_rrsisd_stage2" ;;
    *)
        echo "Invalid dataset/stage combination: ${DATASET:-<unset>}:${STAGE:-<unset>}"
        usage
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAM3_ROOT="${PROJECT_ROOT}/sam3"

echo "========================================"
echo "  RRSIS training"
echo "  Dataset: ${DATASET}"
echo "  Stage:   ${STAGE}"
echo "  GPU:     ${GPU}"
echo "  Config:  ${CONFIG}"
echo "========================================"

cd "${SAM3_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" \
PYTHONPATH="${SAM3_ROOT}:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python sam3/train/train.py \
    -c "${CONFIG}" \
    --use-cluster 0 \
    --num-nodes 1 \
    --num-gpus 1
