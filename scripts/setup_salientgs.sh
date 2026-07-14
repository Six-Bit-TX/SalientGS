#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python - <<'PY'
import shutil
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("A CUDA-enabled PyTorch installation and visible GPU are required.")
if shutil.which("colmap") is None:
    raise SystemExit("COLMAP is required but was not found on PATH.")
print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
PY

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
python -m pip install -e fastmap --no-build-isolation
python -m pip install -e .

export PYTHONPATH="$repo_dir/fastmap${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import torch
import fastmap.cuda
import gsplat
print(f"SalientGS ready (CUDA capability {torch.cuda.get_device_capability()}).")
PY
