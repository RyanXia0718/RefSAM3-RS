#!/usr/bin/env bash

# Unified installation, training, and evaluation entry for RefSAM3-RS.

set -euo pipefail

COMMAND="${1:-help}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
RefSAM3-RS

Usage:
  bash run.sh install
  bash run.sh train --dataset refsegrs|rrsisd --stage 1|2 [--gpu N]
  bash run.sh eval CHECKPOINT [CHECKPOINT ...] \
    --dataset refsegrs|rrsisd [--split val|test] [options]
  bash run.sh inference [inference.py options]
  bash run.sh help

Commands:
  install     Install the tested PyTorch stack and project dependencies.
  train       Run the two-stage full-model training entry.
  eval        Run full-model inference and RRSIS evaluation.
  inference   Run inference only.
  help        Show this message.

Environment:
  Tested on Ubuntu 22.04, Python 3.12, PyTorch 2.7.1, and CUDA 12.8.
  Activate the intended Conda or virtual environment before running install.

Examples:
  conda create -n refsam3_rs python=3.12 -y
  conda activate refsam3_rs
  bash run.sh install

  bash run.sh train --dataset refsegrs --stage 1 --gpu 0
  bash run.sh train --dataset refsegrs --stage 2 --gpu 0
  bash run.sh eval /path/to/checkpoint_80.pt \
    --dataset refsegrs --split test --gpu 0
EOF
}

case "${COMMAND}" in
    install)
        if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
            echo "Warning: no Conda or virtual environment is currently active."
            echo "Activate the intended environment before continuing."
        fi

        echo "Installing PyTorch 2.7.1 with CUDA 12.8 support..."
        python -m pip install \
            torch==2.7.1 \
            torchvision==0.22.1 \
            torchaudio==2.7.1 \
            --index-url https://download.pytorch.org/whl/cu128

        echo "Installing RefSAM3-RS and SAM3 training dependencies..."
        (
            cd "${PROJECT_ROOT}/sam3"
            python -m pip install -e ".[dev,train]"
        )
        python -m pip install pycocotools tqdm pillow einops decord

        echo "Verifying the installation..."
        python -c "import torch, torchvision, sam3; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
        ;;

    train)
        shift
        exec bash "${PROJECT_ROOT}/scripts/train.sh" "$@"
        ;;

    eval)
        shift
        exec bash "${PROJECT_ROOT}/scripts/eval.sh" "$@"
        ;;

    inference)
        shift
        exec python "${PROJECT_ROOT}/scripts/inference.py" "$@"
        ;;

    help|-h|--help)
        usage
        ;;

    *)
        echo "Unknown command: ${COMMAND}" >&2
        usage >&2
        exit 2
        ;;
esac
