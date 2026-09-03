#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/repository"
  exit 2
fi

src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$1" && pwd)"

if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Target is not a Git repository: $repo"
  exit 1
fi

echo "Installing handoff into: $repo"
echo "Current commit: $(git -C "$repo" rev-parse HEAD)"
echo

copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [ -e "$dst" ]; then
    echo "SKIP (exists): $dst"
  else
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
    echo "COPY: $dst"
  fi
}

copy_if_missing "$src_dir/AGENTS.md" "$repo/AGENTS.md"
copy_if_missing "$src_dir/.codex" "$repo/.codex"
copy_if_missing "$src_dir" "$repo/codex_handoff"

echo
echo "Done. Review with:"
echo "  cd \"$repo\""
echo "  git status --short"
echo
echo "Consider adding codex_handoff/ to a personal-only branch if the upstream"
echo "maintainers do not want learning documents in a future PR."
