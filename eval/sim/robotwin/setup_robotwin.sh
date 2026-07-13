#!/usr/bin/env bash
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOTWIN_REPO="${ROBOTWIN_REPO:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_REF="${ROBOTWIN_REF:-main}"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-${SCRIPT_DIR}/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-${SCRIPT_DIR}/robotwin_uv}"
ENV_BACKEND="${ENV_BACKEND:-uv}"
CONDA_ENV="${CONDA_ENV:-embodied-cpp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

SETUP_MODE="${SETUP_MODE:-eval}"
RUN_INSTALL="${RUN_INSTALL:-1}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-1}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-0}"
ALLOW_PARTIAL_ASSETS="${ALLOW_PARTIAL_ASSETS:-0}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/dataset}"
HF_DATASET="${HF_DATASET:-TianxingChen/RoboTwin2.0}"
HF_DATASET_SUBDIR="${HF_DATASET_SUBDIR:-dataset}"
HF_ASSET_MAX_WORKERS="${HF_ASSET_MAX_WORKERS:-1}"
AUTO_PROXY="${AUTO_PROXY:-1}"
PROXY_URL="${PROXY_URL:-}"
TORCH_BACKEND="${TORCH_BACKEND:-cuda}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_CUROBO="${INSTALL_CUROBO:-1}"
CUROBO_REF="${CUROBO_REF:-v0.7.8}"
WARP_LANG_VERSION="${WARP_LANG_VERSION:-1.12.0}"
CUROBO_SETUPTOOLS_VERSION="${CUROBO_SETUPTOOLS_VERSION:-69.5.1}"
INSTALL_OPTIONAL="${INSTALL_OPTIONAL:-0}"
HY_VLA_CLIENT_DEPS="${HY_VLA_CLIENT_DEPS:-transformers==4.51.3 pyzmq protobuf msgpack==1.1.0 msgpack-numpy==0.4.8 scipy}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Clone/setup RoboTwin and optionally download assets / pre-collected dataset.
Default mode installs the simulator/runtime requirements needed for evaluation,
including CUDA PyTorch and RoboTwin's curobo planner dependency. It still skips
training-only extras such as pytorch3d.

Options:
  --repo URL              RoboTwin git repo
                          default: ${ROBOTWIN_REPO}
  --ref REF               branch/tag/commit
                          default: ${ROBOTWIN_REF}
  --dir DIR               checkout directory
                          default: ${ROBOTWIN_DIR}
  --env-backend BACKEND   uv | conda | current
                          default: ${ENV_BACKEND}
  --venv-dir DIR          uv virtualenv parent directory
                          default: ${ROBOTWIN_ENV}
  --conda-env NAME        conda env name
                          default: ${CONDA_ENV}
  --python VERSION        conda python version
                          default: ${PYTHON_VERSION}
  --mode MODE             eval | full
                          eval: requirements + small runtime patches
                          full: run upstream script/_install.sh
                          default: ${SETUP_MODE}
  --no-install            skip Python dependency installation
  --no-assets             skip RoboTwin script/_download_assets.sh
  --allow-partial-assets  continue even if required asset directories are missing
  --download-dataset      download pre-collected trajectories from Hugging Face
  --dataset-dir DIR       destination for pre-collected dataset
                          default: ${DATASET_DIR}
  --proxy URL             proxy URL for git/pip/uv/huggingface downloads
                          e.g. http://127.0.0.1:7890
  --no-auto-proxy         do not auto-detect common local proxy ports
  --torch-backend KIND    cpu | cuda | none
                          cpu: install CPU torch wheels and filter CUDA deps
                          cuda: install CUDA torch wheels from TORCH_CUDA_INDEX_URL
                          none: skip torch install and filter torch deps
                          default: ${TORCH_BACKEND}
  --no-curobo             skip RoboTwin curobo planner installation
  --curobo-ref REF        curobo git ref
                          default: ${CUROBO_REF}
  --with-optional         install optional non-rollout packages from upstream
                          requirements, including azure, wandb, openai
  -h, --help              show this help

Environment overrides:
  ROBOTWIN_REPO, ROBOTWIN_REF, ROBOTWIN_DIR, ROBOTWIN_ENV, ENV_BACKEND,
  CONDA_ENV, PYTHON_VERSION, SETUP_MODE, RUN_INSTALL, DOWNLOAD_ASSETS,
  DOWNLOAD_DATASET, ALLOW_PARTIAL_ASSETS, DATASET_DIR, HF_DATASET,
  HF_DATASET_SUBDIR, HF_ASSET_MAX_WORKERS,
  AUTO_PROXY, PROXY_URL, TORCH_BACKEND, TORCH_VERSION, TORCHVISION_VERSION,
  TORCH_CUDA_INDEX_URL, INSTALL_CUROBO, CUROBO_REF, WARP_LANG_VERSION,
  CUROBO_SETUPTOOLS_VERSION, INSTALL_OPTIONAL
  HY_VLA_CLIENT_DEPS

Notes:
  - Assets are required for simulation/evaluation.
  - Eval mode intentionally skips upstream training-only extras that often make
    setup slow or stuck, such as pytorch3d.
  - The pre-collected trajectory dataset can be large, so it is opt-in.
  - If Hugging Face rate limits asset/dataset downloads, run:
      huggingface-cli login
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) ROBOTWIN_REPO="$2"; shift 2 ;;
        --ref) ROBOTWIN_REF="$2"; shift 2 ;;
        --dir) ROBOTWIN_DIR="$2"; shift 2 ;;
        --env-backend) ENV_BACKEND="$2"; shift 2 ;;
        --venv-dir) ROBOTWIN_ENV="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --mode) SETUP_MODE="$2"; shift 2 ;;
        --no-install) RUN_INSTALL=0; shift ;;
        --no-assets) DOWNLOAD_ASSETS=0; shift ;;
        --allow-partial-assets) ALLOW_PARTIAL_ASSETS=1; shift ;;
        --download-dataset) DOWNLOAD_DATASET=1; shift ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --proxy) PROXY_URL="$2"; shift 2 ;;
        --no-auto-proxy) AUTO_PROXY=0; shift ;;
        --torch-backend) TORCH_BACKEND="$2"; shift 2 ;;
        --no-curobo) INSTALL_CUROBO=0; shift ;;
        --curobo-ref) CUROBO_REF="$2"; shift 2 ;;
        --with-optional) INSTALL_OPTIONAL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "${SETUP_MODE}" in
    eval|full) ;;
    *) echo "ERROR: --mode must be eval or full, got ${SETUP_MODE}" >&2; exit 1 ;;
esac

case "${ENV_BACKEND}" in
    uv|conda|current) ;;
    *) echo "ERROR: --env-backend must be uv, conda, or current, got ${ENV_BACKEND}" >&2; exit 1 ;;
esac

case "${TORCH_BACKEND}" in
    cpu|cuda|none) ;;
    *) echo "ERROR: --torch-backend must be cpu, cuda, or none, got ${TORCH_BACKEND}" >&2; exit 1 ;;
esac

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $1" >&2
        exit 1
    }
}

checkout_ref() {
    local dir="$1"
    local ref="$2"
    git -C "${dir}" fetch origin "${ref}" --depth 1 2>/dev/null || git -C "${dir}" fetch origin --tags --depth 200
    git -C "${dir}" checkout --detach FETCH_HEAD 2>/dev/null || git -C "${dir}" checkout "${ref}"
}

need_cmd git

setup_proxy() {
    if [[ -n "${PROXY_URL}" ]]; then
        export http_proxy="${PROXY_URL}"
        export https_proxy="${PROXY_URL}"
        export HTTP_PROXY="${PROXY_URL}"
        export HTTPS_PROXY="${PROXY_URL}"
        if [[ "${PROXY_URL}" == socks* ]]; then
            export all_proxy="${PROXY_URL}"
            export ALL_PROXY="${PROXY_URL}"
        fi
        echo "[robotwin] using proxy ${PROXY_URL}"
        return
    fi

    if [[ -n "${http_proxy:-}${https_proxy:-}${HTTP_PROXY:-}${HTTPS_PROXY:-}" ]]; then
        echo "[robotwin] using existing proxy environment"
        return
    fi

    if [[ "${AUTO_PROXY}" != "1" ]]; then
        return
    fi

    local port
    for port in 7890 7897 10809 1080; do
        if timeout 1 bash -c ":</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            PROXY_URL="http://127.0.0.1:${port}"
            export http_proxy="${PROXY_URL}"
            export https_proxy="${PROXY_URL}"
            export HTTP_PROXY="${PROXY_URL}"
            export HTTPS_PROXY="${PROXY_URL}"
            echo "[robotwin] auto-detected local proxy ${PROXY_URL}"
            return
        fi
    done
}

setup_proxy

mkdir -p "${SCRIPT_DIR}"

if [[ -d "${ROBOTWIN_DIR}/.git" ]]; then
    echo "[robotwin] updating ${ROBOTWIN_DIR}"
    git -C "${ROBOTWIN_DIR}" remote set-url origin "${ROBOTWIN_REPO}"
    if [[ -n "$(git -C "${ROBOTWIN_DIR}" status --porcelain)" ]]; then
        echo "[robotwin] checkout has local changes; keeping existing tree"
    else
        checkout_ref "${ROBOTWIN_DIR}" "${ROBOTWIN_REF}"
    fi
elif [[ -e "${ROBOTWIN_DIR}" ]]; then
    echo "ERROR: ${ROBOTWIN_DIR} exists but is not a git checkout" >&2
    exit 1
else
    echo "[robotwin] cloning ${ROBOTWIN_REPO} -> ${ROBOTWIN_DIR}"
    git clone --depth 1 "${ROBOTWIN_REPO}" "${ROBOTWIN_DIR}"
    checkout_ref "${ROBOTWIN_DIR}" "${ROBOTWIN_REF}"
fi

case "${ENV_BACKEND}" in
    uv)
        need_cmd uv
        mkdir -p "${ROBOTWIN_ENV}"
        if [[ "${RUN_INSTALL}" == "0" && -x "${ROBOTWIN_ENV}/.venv/bin/python" ]]; then
            echo "[robotwin] reusing uv venv ${ROBOTWIN_ENV}/.venv"
        else
            rm -rf "${ROBOTWIN_ENV}/.venv"
            echo "[robotwin] creating uv venv ${ROBOTWIN_ENV}/.venv python=${PYTHON_VERSION}"
            uv venv "${ROBOTWIN_ENV}/.venv" --python "${PYTHON_VERSION}"
        fi
        # shellcheck disable=SC1091
        source "${ROBOTWIN_ENV}/.venv/bin/activate"
        ;;
    conda)
        need_cmd conda
        export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"
        eval "$(conda shell.bash hook)"
        if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
            echo "[robotwin] creating conda env ${CONDA_ENV} python=${PYTHON_VERSION}"
            conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
        fi
        conda activate "${CONDA_ENV}"
        ;;
    current)
        echo "[robotwin] using current Python environment"
        ;;
esac

echo "[robotwin] python=$(command -v python)"
python --version

cd "${ROBOTWIN_DIR}"

if [[ "${RUN_INSTALL}" == "1" ]]; then
    if [[ "${SETUP_MODE}" == "full" ]]; then
        if [[ ! -f script/_install.sh ]]; then
            echo "ERROR: script/_install.sh not found in ${ROBOTWIN_DIR}" >&2
            exit 1
        fi
        echo "[robotwin] running full upstream install script/_install.sh"
        bash script/_install.sh
    else
        if [[ ! -f script/requirements.txt ]]; then
            echo "ERROR: script/requirements.txt not found in ${ROBOTWIN_DIR}" >&2
            exit 1
        fi
        echo "[robotwin] installing eval/runtime requirements from script/requirements.txt"
        req_file="script/requirements.txt"
        if command -v uv >/dev/null 2>&1; then
            uv pip install -U pip
        else
            python -m ensurepip --upgrade
            python -m pip install -U pip
        fi
        if command -v uv >/dev/null 2>&1; then
            uv pip install -U "setuptools<81"
        else
            python -m pip install -U "setuptools<81"
        fi
        if [[ "${TORCH_BACKEND}" == "cpu" ]]; then
            echo "[robotwin] installing CPU torch wheels torch=${TORCH_VERSION} torchvision=${TORCHVISION_VERSION}"
            if command -v uv >/dev/null 2>&1; then
                uv pip install \
                    "torch==${TORCH_VERSION}+cpu" \
                    "torchvision==${TORCHVISION_VERSION}+cpu" \
                    --index-url https://download.pytorch.org/whl/cpu
            else
                python -m pip install \
                    "torch==${TORCH_VERSION}+cpu" \
                    "torchvision==${TORCHVISION_VERSION}+cpu" \
                    --index-url https://download.pytorch.org/whl/cpu
            fi
        elif [[ "${TORCH_BACKEND}" == "cuda" ]]; then
            echo "[robotwin] installing CUDA torch wheels torch=${TORCH_VERSION} torchvision=${TORCHVISION_VERSION}"
            if command -v uv >/dev/null 2>&1; then
                uv pip install \
                    "torch==${TORCH_VERSION}" \
                    "torchvision==${TORCHVISION_VERSION}" \
                    --index-url "${TORCH_CUDA_INDEX_URL}"
            else
                python -m pip install \
                    "torch==${TORCH_VERSION}" \
                    "torchvision==${TORCHVISION_VERSION}" \
                    --index-url "${TORCH_CUDA_INDEX_URL}"
            fi
        fi
        if [[ "${TORCH_BACKEND}" == "cpu" || "${TORCH_BACKEND}" == "cuda" || "${TORCH_BACKEND}" == "none" ]]; then
            req_file="$(mktemp)"
            grep -v -E '^(torch|torchvision|torchaudio|nvidia-)' script/requirements.txt > "${req_file}"
            echo "[robotwin] filtered torch/CUDA packages from requirements (${TORCH_BACKEND} backend)"
        fi
        if [[ "${INSTALL_OPTIONAL}" != "1" ]]; then
            filtered_req="$(mktemp)"
            grep -v -E '^(azure([-_.A-Za-z0-9]*)?|wandb|openai([-_.A-Za-z0-9]*)?)([<=> ].*)?$' "${req_file}" > "${filtered_req}"
            req_file="${filtered_req}"
            echo "[robotwin] filtered optional non-rollout packages; pass --with-optional to install them"
        fi
        if command -v uv >/dev/null 2>&1; then
            uv pip install -r "${req_file}"
        else
            python -m pip install -r "${req_file}"
        fi

        echo "[robotwin] installing HY-VLA C++ client/eval dependencies"
        if command -v uv >/dev/null 2>&1; then
            # shellcheck disable=SC2086
            uv pip install ${HY_VLA_CLIENT_DEPS}
        else
            # shellcheck disable=SC2086
            python -m pip install ${HY_VLA_CLIENT_DEPS}
        fi

        echo "[robotwin] applying upstream runtime patches for sapien/mplib if installed"
        SAPIEN_LOCATION="$(python - <<'PY'
import pathlib
try:
    import sapien
    print(pathlib.Path(sapien.__file__).resolve().parent)
except Exception:
    print("")
PY
)"
        if [[ -n "${SAPIEN_LOCATION}" && -f "${SAPIEN_LOCATION}/wrapper/urdf_loader.py" ]]; then
            sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
        else
            echo "[robotwin] warning: sapien wrapper/urdf_loader.py not found; skipping sapien patch"
        fi

        MPLIB_LOCATION="$(python - <<'PY'
import pathlib
try:
    import mplib
    print(pathlib.Path(mplib.__file__).resolve().parent)
except Exception:
    print("")
PY
)"
        if [[ -n "${MPLIB_LOCATION}" && -f "${MPLIB_LOCATION}/planner.py" ]]; then
            sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${MPLIB_LOCATION}/planner.py"
        else
            echo "[robotwin] warning: mplib planner.py not found; skipping mplib patch"
        fi

        if [[ "${INSTALL_CUROBO}" == "1" ]]; then
            if [[ "${TORCH_BACKEND}" != "cuda" ]]; then
                echo "ERROR: curobo requires CUDA torch. Use --torch-backend cuda or pass --no-curobo." >&2
                exit 1
            fi
            python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA torch is installed but torch.cuda.is_available() is false")
print(f"[robotwin] verified CUDA torch: {torch.__version__}, cuda={torch.version.cuda}, device={torch.cuda.get_device_name(0)}")
PY
            echo "[robotwin] installing curobo ${CUROBO_REF}"
            mkdir -p envs
            if [[ -d envs/curobo/.git ]]; then
                git -C envs/curobo fetch origin "${CUROBO_REF}" --depth 1 || true
                git -C envs/curobo checkout "${CUROBO_REF}" 2>/dev/null || git -C envs/curobo checkout --detach FETCH_HEAD
            else
                git clone --branch "${CUROBO_REF}" --depth 1 https://github.com/NVlabs/curobo.git envs/curobo
            fi
            (
                cd envs/curobo
                python -m pip install -e . --no-build-isolation
            )
            python -m pip install "warp-lang==${WARP_LANG_VERSION}" "setuptools==${CUROBO_SETUPTOOLS_VERSION}"
        else
            echo "[robotwin] skipping curobo install"
        fi

        echo "[robotwin] verifying runtime imports"
        python - <<'PY'
import importlib
mods = ["yaml", "numpy", "scipy", "zmq", "google.protobuf", "msgpack", "msgpack_numpy", "transformers", "sapien", "mplib"]
missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("ERROR: missing runtime imports:\n  " + "\n  ".join(missing))
try:
    import torch
    print(f"[robotwin] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    raise SystemExit(f"ERROR: torch import failed: {exc}")
PY
    fi
else
    echo "[robotwin] skipping install"
fi

if [[ "${DOWNLOAD_ASSETS}" == "1" ]]; then
    if [[ ! -f script/_download_assets.sh ]]; then
        echo "ERROR: script/_download_assets.sh not found in ${ROBOTWIN_DIR}" >&2
        exit 1
    fi
    if ! python - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
    then
        echo "[robotwin] installing huggingface_hub for asset download"
        if command -v uv >/dev/null 2>&1; then
            uv pip install "huggingface_hub[cli]"
        else
            python -m pip install "huggingface_hub[cli]"
        fi
    fi
    need_cmd unzip
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
    export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
    echo "[robotwin] downloading assets from ${HF_DATASET} with max_workers=${HF_ASSET_MAX_WORKERS}"
    (
        cd assets
        python - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_DATASET}",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
    max_workers=int(os.environ.get("HF_ASSET_MAX_WORKERS", "${HF_ASSET_MAX_WORKERS}")),
)
PY
        for asset_zip in background_texture.zip embodiments.zip objects.zip; do
            if [[ ! -f "${asset_zip}" ]]; then
                echo "ERROR: asset zip not found after download: assets/${asset_zip}" >&2
                exit 1
            fi
            unzip -oq "${asset_zip}"
            rm -f "${asset_zip}"
        done
    )
    echo "[robotwin] configuring asset paths"
    python ./script/update_embodiment_config_path.py
    missing_assets=()
    for required_asset_dir in assets/embodiments assets/objects assets/background_texture; do
        if [[ ! -d "${required_asset_dir}" ]]; then
            missing_assets+=("${required_asset_dir}")
        fi
    done
    if [[ "${#missing_assets[@]}" -gt 0 ]]; then
        echo "[robotwin] warning: missing required asset directories:" >&2
        printf '  %s\n' "${missing_assets[@]}" >&2
        echo "[robotwin] Hugging Face/Xet downloads may have timed out even though the upstream script returned success." >&2
        echo "[robotwin] rerun this script, or pass --allow-partial-assets only if you know your target task does not need them." >&2
        if [[ "${ALLOW_PARTIAL_ASSETS}" != "1" ]]; then
            exit 1
        fi
    fi
else
    echo "[robotwin] skipping asset download"
fi

if [[ "${DOWNLOAD_DATASET}" == "1" ]]; then
    mkdir -p "${DATASET_DIR}"
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        python -m pip install -U "huggingface_hub[cli]"
    fi
    echo "[robotwin] downloading pre-collected dataset ${HF_DATASET}/${HF_DATASET_SUBDIR} -> ${DATASET_DIR}"
    huggingface-cli download "${HF_DATASET}" \
        --repo-type dataset \
        --include "${HF_DATASET_SUBDIR}/**" \
        --local-dir "${DATASET_DIR}"
else
    echo "[robotwin] skipping pre-collected dataset download"
fi

cat <<EOF
[robotwin] setup complete
Mode:
  ${SETUP_MODE}
Environment backend:
  ${ENV_BACKEND}

RoboTwin checkout:
  ${ROBOTWIN_DIR}
RobotWin Python:
  $(command -v python)

To run HY-VLA RobotWin evaluation from this environment:
  cd ${REPO_ROOT}
  cmake -S . -B build \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DMODEL_BUILD_VLA_HY_VLA=ON \\
    -DGGML_CUDA=ON \\
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \\
    -DCMAKE_CUDA_ARCHITECTURES=<your-arch>
  cmake --build build --target vla-hy-vla-server -j"\$(nproc)"
  export GGML_CUDA_DISABLE_GRAPHS=1
  \$(command -v python) \\
    eval/client/run_robotwin_eval.py \\
    --model /path/to/Hy-Embodied-0.5-VLA-RoboTwin_bf16.gguf \\
    --tokenizer /path/to/HY-VLA \\
    --episodes 1 \\
    --max-steps 0

Generic gym-style adapter, if your RoboTwin env is registered through gym:
  ROBOTWIN_PYTHON=$(command -v python) \\
  bash eval/run_robotwin_hy_vla_client.sh \\
    -e <robotwin_gym_env_id> \\
    -t /path/to/hy_vla_checkpoint_or_tokenizer \\
    -p ${ROBOTWIN_DIR} \\
    -m <robotwin_registration_module>
EOF
