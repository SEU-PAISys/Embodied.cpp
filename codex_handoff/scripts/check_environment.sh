#!/usr/bin/env bash
set -u

echo "=== Timestamp ==="
date -Is 2>/dev/null || date
echo

echo "=== OS / WSL ==="
uname -a
if [ -f /etc/os-release ]; then
  cat /etc/os-release
fi
echo

echo "=== CPU / Memory / Disk ==="
command -v lscpu >/dev/null && lscpu | sed -n '1,25p'
command -v free >/dev/null && free -h
df -h "$HOME"
echo

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
else
  echo "nvidia-smi: NOT FOUND"
fi
echo

echo "=== Compilers / Build Tools ==="
for cmd in gcc g++ cmake ninja git protoc pkg-config; do
  if command -v "$cmd" >/dev/null; then
    echo "--- $cmd ---"
    "$cmd" --version 2>&1 | head -n 3
  else
    echo "$cmd: NOT FOUND"
  fi
done
echo

echo "=== CUDA Toolkit ==="
if command -v nvcc >/dev/null; then
  nvcc --version
else
  echo "nvcc: NOT FOUND (this is acceptable before CUDA compilation)"
fi
echo

echo "=== Python ==="
command -v python3 || true
python3 --version 2>/dev/null || true
command -v python || true
python --version 2>/dev/null || true
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<not active>}"
echo

echo "=== PyTorch (active environment only) ==="
python - <<'PY' 2>/dev/null || echo "PyTorch unavailable in current Python environment"
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
echo

echo "=== Git repository (if any) ==="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "root: $(git rev-parse --show-toplevel)"
  echo "commit: $(git rev-parse HEAD)"
  echo "branch: $(git branch --show-current)"
  git status --short
else
  echo "Not inside a Git repository"
fi
