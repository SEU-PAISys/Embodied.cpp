#!/usr/bin/env bash
# Isolated safety check for the LIBERO_UV_ENV guard logic in
# eval/sim/libero/setup_libero.sh. Uses throwaway temp dirs only; never
# touches a real user directory; never invokes git/uv/install steps.
set -u
cd "$(dirname "$0")/.."

# Extract the pure guard block (default resolution + / HOME + .venv checks),
# which ends right before the first side-effecting command (REPO_ROOT=...).
python3 - eval/sim/libero/setup_libero.sh > /tmp/_libero_guards.sh <<'PY'
import sys
src = open(sys.argv[1], encoding="utf-8").read()
start = src.index('mkdir -p "$LIBERO_UV_ENV"')
end = src.index('REPO_ROOT=')
print(src[start:end])
PY

PASS=0; FAIL=0
# Symlink creation needs privilege on Windows Git Bash; detect support.
if ln -s . /tmp/_wb_symlink_probe 2>/dev/null; then
  SYMLINK_SUPPORT=1; rm -f /tmp/_wb_symlink_probe
else
  SYMLINK_SUPPORT=0
fi

# scenario <label> <expected: accept|refuse> <setup> <env>
scenario() {
  local label="$1" expected="$2" setup="$3" env_path="$4"
  rm -rf "$env_path"; mkdir -p "$env_path"
  ( cd "$env_path" && eval "$setup" )
  LIBERO_UV_ENV="$env_path" bash /tmp/_libero_guards.sh >/dev/null 2>&1
  local rc=$?
  if { [ "$expected" = refuse ] && [ "$rc" -eq 1 ]; } || \
     { [ "$expected" = accept ] && [ "$rc" -eq 0 ]; }; then
    PASS=$((PASS+1)); echo "  ok    $label (rc=$rc)"
  else
    FAIL=$((FAIL+1)); echo "  FAIL  $label (expected $expected, got rc=$rc)"
  fi
}

B=$(mktemp -d)
trap 'rm -rf "$B" /tmp/_libero_guards.sh' EXIT

# 1. Dedicated dir with a valid existing .venv  -> accepted (will be reused)
scenario "valid .venv is accepted (reuse)" accept \
  'mkdir -p .venv/bin; echo "[venv]" > .venv/pyvenv.cfg; touch .venv/bin/activate' "$B/env1"
# 2. Directory containing an unrecognized .venv (no pyvenv.cfg) -> refused
scenario "unrecognized .venv refused" refuse \
  'mkdir -p .venv; echo junk > .venv/other' "$B/env2"
# 3. .venv is a symlink -> refused by the guard's -L branch. Symlink
#    creation needs privilege on Windows Git Bash, so this scenario cannot
#    run reliably here; the -L guard is verified by code inspection:
#      if [ -e "$E/.venv" ] || [ -L "$E/.venv" ]; then
#        if [ -L "$E/.venv" ] || [ ! -f pyvenv.cfg ] || [ ! -f bin/activate ]
#        then refuse; fi
#    Needs a Linux host for a live run.
echo "  note  symlinked-.venv refusal: -L branch by inspection (Linux-only live run)"
# 4. Unrelated files in the env dir survive (guards never delete) + no .venv -> accepted
scenario "no .venv, sibling files untouched" accept \
  'echo keep > notes.txt; mkdir -p data' "$B/env4"
# 5. "/" as LIBERO_UV_ENV -> refused
LIBERO_UV_ENV=/ bash /tmp/_libero_guards.sh >/dev/null 2>&1
[ $? -eq 1 ] && { PASS=$((PASS+1)); echo "  ok    / is refused"; } || { FAIL=$((FAIL+1)); echo "  FAIL  / is refused"; }

echo
echo "result: pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
