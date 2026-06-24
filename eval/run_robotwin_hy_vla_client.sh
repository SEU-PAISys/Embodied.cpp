#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") -e ENV_ID -t TOKENIZER [-a ADDR] [-o OUTPUT_ROOT] [-n N_EPISODES]

  -e ENV_ID        RobotWin gymnasium env id.
  -t TOKENIZER     HY-VLA tokenizer/checkpoint directory.
  -a ADDR          vla-server address (default: tcp://localhost:5555).
  -o OUTPUT_ROOT   output directory (default: ${REPO_ROOT}/outputs/robotwin_hy_vla).
  -n N_EPISODES    episodes (default: 10).
  -m MODULE        module to import before gym.make; repeatable.
  -p PYTHONPATH    path inserted into Python sys.path before module import; repeatable.
  -h               help.

Extra env overrides:
  ROBOTWIN_PYTHON, MAX_STEPS, STATE_DIM, ACTION_DIM, N_ACTION_STEPS,
  FRONT_KEY, WRIST_KEY, STATE_KEY, TASK_KEY, TASK
EOF
}

ENV_ID=""
TOKENIZER=""
ADDR="${CLIENT_ADDR:-tcp://localhost:5555}"
OUTPUT_ROOT="${REPO_ROOT}/outputs/robotwin_hy_vla"
N_EPISODES="10"
REGISTER_MODULES=()
PYTHONPATHS=()

# Keep local 8GB laptop GPUs on the stable ggml CUDA path by default.
# Override with GGML_CUDA_DISABLE_GRAPHS=0 only on GPUs with enough VRAM.
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

while getopts ":e:t:a:o:n:m:p:h" opt; do
    case "${opt}" in
        e) ENV_ID="${OPTARG}" ;;
        t) TOKENIZER="${OPTARG}" ;;
        a) ADDR="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        n) N_EPISODES="${OPTARG}" ;;
        m) REGISTER_MODULES+=("${OPTARG}") ;;
        p) PYTHONPATHS+=("${OPTARG}") ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${ENV_ID}" || -z "${TOKENIZER}" ]]; then
    usage >&2
    exit 1
fi

case "${ADDR}" in
    tcp://*) ;;
    *:*) ADDR="tcp://${ADDR}" ;;
    *) ADDR="tcp://${ADDR}:5555" ;;
esac

DEFAULT_ROBOTWIN_PY="${REPO_ROOT}/eval/sim/robotwin/robotwin_uv/.venv/bin/python"
if [[ -z "${ROBOTWIN_PYTHON:-}" && -x "${DEFAULT_ROBOTWIN_PY}" ]]; then
    PY="${DEFAULT_ROBOTWIN_PY}"
else
    PY="${ROBOTWIN_PYTHON:-python3}"
fi
CLIENT="${REPO_ROOT}/eval/client/run_robotwin_eval.py"

args=(
    "${CLIENT}"
    --env-id "${ENV_ID}"
    --tokenizer "${TOKENIZER}"
    --vla-addr "${ADDR}"
    --output-dir "${OUTPUT_ROOT}"
    --n-episodes "${N_EPISODES}"
    --state-dim "${STATE_DIM:-20}"
    --action-dim "${ACTION_DIM:-20}"
    --n-action-steps "${N_ACTION_STEPS:-8}"
)

[[ -n "${MAX_STEPS:-}" ]] && args+=(--max-steps "${MAX_STEPS}")
[[ -n "${FRONT_KEY:-}" ]] && args+=(--front-key "${FRONT_KEY}")
[[ -n "${WRIST_KEY:-}" ]] && args+=(--wrist-key "${WRIST_KEY}")
[[ -n "${STATE_KEY:-}" ]] && args+=(--state-key "${STATE_KEY}")
[[ -n "${TASK_KEY:-}" ]] && args+=(--task-key "${TASK_KEY}")
[[ -n "${TASK:-}" ]] && args+=(--task "${TASK}")

for p in "${PYTHONPATHS[@]}"; do
    args+=(--robotwin-pythonpath "${p}")
done
for m in "${REGISTER_MODULES[@]}"; do
    args+=(--register-module "${m}")
done

echo "[robotwin] python=${PY}"
echo "[robotwin] env=${ENV_ID}"
echo "[robotwin] server=${ADDR}"
"${PY}" "${args[@]}"
