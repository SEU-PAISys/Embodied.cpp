#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <build-dir> <target> [log-file]"
  exit 2
fi

build_dir="$1"
target="$2"
log_file="${3:-build_${target}_$(date +%Y%m%d_%H%M%S).log}"

echo "Build directory: $build_dir"
echo "Target: $target"
echo "Log: $log_file"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo N/A)"
echo

set +e
cmake --build "$build_dir" --target "$target" -j1 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e

echo
echo "Build exit code: $status"
echo "First compiler-style errors:"
grep -nEi '(^|[[:space:]])(fatal error:|error:|undefined reference|CMake Error)' "$log_file" | head -n 20 || true

exit "$status"
