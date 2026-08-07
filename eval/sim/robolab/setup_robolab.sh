#!/usr/bin/env bash
# Copyright 2026 SEU-PAISys
# SPDX-License-Identifier: Apache-2.0

# Install the host-side RoboLab environment used by the Cosmos3 PyTorch
# integration.  Isaac Sim is intentionally installed outside the repository's
# tracked source tree and is ignored by .gitignore.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOLAB_GIT_URL="${ROBOLAB_GIT_URL:-https://github.com/NVlabs/RoboLab.git}"
ROBOLAB_DIR="${ROBOLAB_DIR:-$SCRIPT_DIR/RoboLab}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
ROBOLAB_EXTRA="${ROBOLAB_EXTRA:-isaac50}"
ROBOLAB_GPU="${ROBOLAB_GPU:-}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"

export UV_CACHE_DIR
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then return; fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then export PATH="$HOME/.local/bin:$PATH"; return; fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

ensure_checkout() {
    mkdir -p "$(dirname "$ROBOLAB_DIR")"
    if [[ -d "$ROBOLAB_DIR/.git" ]]; then return; fi
    if [[ -d "$ROBOLAB_DIR" && -n "$(ls -A "$ROBOLAB_DIR" 2>/dev/null)" ]]; then
        echo "ROBOLAB_DIR exists but is not a RoboLab checkout: $ROBOLAB_DIR" >&2
        exit 2
    fi
    git clone "$ROBOLAB_GIT_URL" "$ROBOLAB_DIR"
}

write_isaaclab_path_shim() {
    local site_dir base
    site_dir="$($ROBOLAB_DIR/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
    base="$site_dir/isaaclab/source"
    printf '%s\n' "import sys; [sys.path.insert(0, p) for p in [\"$base/isaaclab\", \"$base/isaaclab_assets\", \"$base/isaaclab_tasks\", \"$base/isaaclab_mimic\", \"$base/isaaclab_rl\"] if p not in sys.path]" > "$site_dir/isaaclab_source_paths.pth"
}

run_smoke() {
    [[ -n "$ROBOLAB_GPU" ]] || { echo "Set ROBOLAB_GPU before running the smoke test." >&2; exit 2; }
    (
        cd "$ROBOLAB_DIR"
        CUDA_VISIBLE_DEVICES="$ROBOLAB_GPU" timeout "${ROBOLAB_SMOKE_TIMEOUT:-420}" .venv/bin/python - <<'PY'
import argparse
import cv2  # noqa: F401; must precede isaaclab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--device", "cuda:0"])
args.enable_cameras = True
app = AppLauncher(args).app

from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode

auto_register_droid_envs(task=["BananaInBowlTask"])
env, _ = create_env(get_envs(task=["BananaInBowlTask"])[0], device="cuda:0", num_envs=1, use_fabric=True)
env.reset()
end_episode(env)
env.close()
app.close()
print("ROBOLAB_SMOKE_OK")
PY
    )
}

ensure_uv
ensure_checkout
cd "$ROBOLAB_DIR"
uv python install 3.11
[[ -x .venv/bin/python ]] || uv venv --python 3.11
uv sync --extra "$ROBOLAB_EXTRA"
uv pip install --index-url https://pypi.org/simple openpi-client
write_isaaclab_path_shim

if [[ "$RUN_SMOKE_TEST" == "1" ]]; then run_smoke; fi
