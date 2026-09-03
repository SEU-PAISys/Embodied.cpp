#!/usr/bin/env bash
set -euo pipefail

out="${1:-repo_snapshot_$(date +%Y%m%d_%H%M%S).txt}"

{
  echo "timestamp=$(date -Is 2>/dev/null || date)"
  echo "repo=$(git rev-parse --show-toplevel)"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo
  echo "=== status ==="
  git status --short
  echo
  echo "=== submodules ==="
  git submodule status || true
  echo
  echo "=== recent commits ==="
  git log -n 5 --oneline
} > "$out"

echo "Saved $out"
