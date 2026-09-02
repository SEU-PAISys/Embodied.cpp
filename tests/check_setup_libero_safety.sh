#!/usr/bin/env bash
# Isolated safety check for the LIBERO_UV_ENV guard logic in
# eval/sim/libero/setup_libero.sh. Uses throwaway temp dirs only; never
# touches a real user directory; never invokes git/uv/install steps.
set -u
cd "$(dirname "$0")/.."

# Everything this script touches lives under one unique temp dir (guard
# fragment, symlink probe and scenario envs), so parallel runs never collide
# and the exit cleanup only removes our own directory. Abort immediately if
# the directory cannot be created - with an empty SAFE_TMP every later
# rm/mkdir would fall back to the filesystem root and defeat the isolation.
# The variable is deliberately non-generic so it cannot shadow TMP/TEMP.
SAFE_TMP="$(mktemp -d 2>/dev/null)" || {
    echo "check_setup_libero_safety: cannot create a temporary directory" >&2
    exit 1
}
[ -d "$SAFE_TMP" ] || {
    echo "check_setup_libero_safety: mktemp did not produce a usable directory" >&2
    exit 1
}
GUARD="$SAFE_TMP/guards.sh"
PROBE="$SAFE_TMP/symlink_probe"
trap 'rm -rf "$SAFE_TMP"' EXIT

# Extract the pure guard block (default resolution + / HOME + .venv checks),
# which ends right before the first side-effecting command (REPO_ROOT=...).
python3 - eval/sim/libero/setup_libero.sh > "$GUARD" <<'PY'
import sys
src = open(sys.argv[1], encoding="utf-8").read()
start = src.index('mkdir -p "$LIBERO_UV_ENV"')
end = src.index('REPO_ROOT=')
print(src[start:end])
PY

PASS=0; FAIL=0; SKIP=0
# Symlink creation needs privilege on Windows Git Bash, and MSYS ln -s can
# silently "succeed" without creating anything. Detect real support by
# checking the link actually exists.
rm -f "$PROBE"
if ln -s . "$PROBE" 2>/dev/null && [ -L "$PROBE" ]; then
  SYMLINK_SUPPORT=1; rm -f "$PROBE"
else
  SYMLINK_SUPPORT=0; rm -f "$PROBE"
fi

# scenario <label> <expected: accept|refuse> <setup> <env>
scenario() {
  local label="$1" expected="$2" setup="$3" env_path="$4"
  rm -rf "$env_path"; mkdir -p "$env_path"
  ( cd "$env_path" && eval "$setup" )
  local setup_rc=$?
  if [ "$setup_rc" -ne 0 ]; then
    SKIP=$((SKIP+1)); echo "  skip  $label (setup failed rc=$setup_rc)"
    return
  fi
  LIBERO_UV_ENV="$env_path" bash "$GUARD" >/dev/null 2>&1
  local rc=$?
  if { [ "$expected" = refuse ] && [ "$rc" -eq 1 ]; } || \
     { [ "$expected" = accept ] && [ "$rc" -eq 0 ]; }; then
    PASS=$((PASS+1)); echo "  ok    $label (rc=$rc)"
  else
    FAIL=$((FAIL+1)); echo "  FAIL  $label (expected $expected, got rc=$rc)"
  fi
}

# Scenario envs live under the same unique temp dir; the single EXIT trap
# cleans everything this script created.
B="$SAFE_TMP/scenarios"
mkdir -p "$B"

# 1. Dedicated dir with a valid existing .venv  -> accepted (will be reused)
scenario "valid .venv is accepted (reuse)" accept \
  'mkdir -p .venv/bin; echo "[venv]" > .venv/pyvenv.cfg; touch .venv/bin/activate' "$B/env1"
# 2. Directory containing an unrecognized .venv (no pyvenv.cfg) -> refused
scenario "unrecognized .venv refused" refuse \
  'mkdir -p .venv; echo junk > .venv/other' "$B/env2"
# 3. .venv is a symlink -> refused by the guard's -L branch. Runs for real
#    when the host can create symlinks; otherwise marked skipped. Kept out of
#    the generic scenario() because MSYS ln -s can silently no-op.
if [ "$SYMLINK_SUPPORT" -eq 1 ]; then
  env3="$B/env3"; mkdir -p "$env3" "$B/realvenv"
  ( cd "$env3" && ln -s ../realvenv .venv )
  if [ -L "$env3/.venv" ]; then
    LIBERO_UV_ENV="$env3" bash "$GUARD" >/dev/null 2>&1
    if [ $? -eq 1 ]; then
      PASS=$((PASS+1)); echo "  ok    symlinked .venv refused (rc=1)"
    else
      FAIL=$((FAIL+1)); echo "  FAIL  symlinked .venv was not refused"
    fi
  else
    SKIP=$((SKIP+1)); echo "  skip  symlinked .venv refused (link not created despite probe)"
  fi
else
  SKIP=$((SKIP+1)); echo "  skip  symlinked .venv refused (host cannot create symlinks; -L branch by inspection)"
fi
# 4. No .venv: guards must accept the dir AND leave sibling files intact.
scenario "no .venv accepted, sibling files untouched" accept \
  'echo keep > notes.txt; mkdir -p data' "$B/env4"
if [ -f "$B/env4/notes.txt" ] && [ "$(cat "$B/env4/notes.txt")" = keep ] \
   && [ -d "$B/env4/data" ]; then
  PASS=$((PASS+1)); echo "  ok    sibling files survived unchanged (notes.txt + data/)"
else
  FAIL=$((FAIL+1)); echo "  FAIL  sibling files were deleted or modified"
fi
# 5. "/" as LIBERO_UV_ENV -> refused
LIBERO_UV_ENV=/ bash "$GUARD" >/dev/null 2>&1
[ $? -eq 1 ] && { PASS=$((PASS+1)); echo "  ok    / is refused"; } || { FAIL=$((FAIL+1)); echo "  FAIL  / is refused"; }
# 6. HOME as LIBERO_UV_ENV -> refused
LIBERO_UV_ENV="$HOME" bash "$GUARD" >/dev/null 2>&1
[ $? -eq 1 ] && { PASS=$((PASS+1)); echo "  ok    \$HOME is refused"; } || { FAIL=$((FAIL+1)); echo "  FAIL  \$HOME is refused"; }

echo
echo "result: pass=$PASS fail=$FAIL skip=$SKIP"
[ "$FAIL" -eq 0 ]
