#!/usr/bin/env bash

# Minimal full-model evaluation entry for RefSegRS and RRSIS-D.

set -euo pipefail

DATASET=""
SPLIT="test"
GPU="0"
BATCH_SIZE="1"
NUM_WORKERS="2"
OUT_DIR=""
NO_INFER=0
CHECKPOINTS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval.sh CHECKPOINT [CHECKPOINT ...] \
    --dataset refsegrs|rrsisd [--split val|test] [options]

Options:
  --gpu N
  --batch-size N
  --num-workers N
  --out-dir DIR
  --no-infer
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2"; shift 2 ;;
        --split)       SPLIT="$2"; shift 2 ;;
        --gpu)         GPU="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2"; shift 2 ;;
        --no-infer)    NO_INFER=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        --*)           echo "Unknown option: $1"; usage; exit 2 ;;
        *)             CHECKPOINTS+=("$1"); shift ;;
    esac
done

if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "Invalid split: $SPLIT"
    usage
    exit 2
fi
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "At least one checkpoint is required."
    usage
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "$DATASET" in
    refsegrs)
        GT_JSON="${PROJECT_ROOT}/datasets/RefSegRS/sam3i_${SPLIT}.json"
        IMAGE_ROOT="${PROJECT_ROOT}/datasets/RefSegRS/images"
        ;;
    rrsisd)
        GT_JSON="${PROJECT_ROOT}/datasets/RRSIS-D/sam3i_${SPLIT}.json"
        IMAGE_ROOT="${PROJECT_ROOT}/datasets/RRSIS-D/images/rrsisd/JPEGImages"
        ;;
    *)
        echo "Invalid dataset: ${DATASET:-<unset>}"
        usage
        exit 2
        ;;
esac

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/eval_${DATASET}_full_${SPLIT}}"
mkdir -p "$OUT_DIR"

echo "========================================"
echo "  RRSIS full-model evaluation"
echo "  Dataset:    $DATASET"
echo "  Split:      $SPLIT"
echo "  GPU:        $GPU"
echo "  Batch size: $BATCH_SIZE"
echo "  GT:         $GT_JSON"
echo "  Images:     $IMAGE_ROOT"
echo "  Output:     $OUT_DIR"
echo "========================================"

TABLE_ROWS=(
    "| Checkpoint | gIoU | cIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |"
    "|---|---:|---:|---:|---:|---:|---:|---:|"
)

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_NAME="$(basename "$CHECKPOINT" .pt)"
    PREDICTIONS="${OUT_DIR}/pred_${CHECKPOINT_NAME}.json"
    METRICS="${OUT_DIR}/metrics_${CHECKPOINT_NAME}.json"

    echo
    echo "Checkpoint: $CHECKPOINT_NAME"

    if [[ $NO_INFER -eq 0 ]]; then
        python "${SCRIPT_DIR}/inference.py" \
            --annotations "$GT_JSON" \
            --image-root "$IMAGE_ROOT" \
            --checkpoint "$CHECKPOINT" \
            --output "$PREDICTIONS" \
            --gpu "$GPU" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS"
    elif [[ ! -f "$PREDICTIONS" ]]; then
        echo "Prediction file not found: $PREDICTIONS"
        exit 1
    fi

    python "${SCRIPT_DIR}/evaluate.py" \
        --gt "$GT_JSON" \
        --predictions "$PREDICTIONS" \
        --output-json "$METRICS"

    ROW="$(python -c 'import json, sys; m=json.load(open(sys.argv[1])); print("| " + sys.argv[2] + " | " + " | ".join(f"{m[k]:.2f}%" for k in ["gIoU", "cIoU", "P@0.5", "P@0.6", "P@0.7", "P@0.8", "P@0.9"]) + " |")' "$METRICS" "$CHECKPOINT_NAME")"
    TABLE_ROWS+=("$ROW")
done

echo
echo "========================================"
echo "  Summary"
echo "========================================"
printf '%s\n' "${TABLE_ROWS[@]}"
