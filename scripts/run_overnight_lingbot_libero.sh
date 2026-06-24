#!/usr/bin/env bash
set -u

ROOT="/home/xuling/robotic_code/embodied.cpp/embodied.cpp-main-backup-20260621-light"
PY="$ROOT/eval/sim/libero/libero_uv/.venv/bin/python"
SERVER="$ROOT/build/lingbot-world-server"
MODEL="/home/xuling/robotic_dataset/models/lingbot_va_transformer_wan_q4_K.gguf"
TOKENIZER="/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer"
ADDR="tcp://127.0.0.1:5557"
SUMMARY="$ROOT/outputs/overnight_lingbot_va_q4k_summary.md"

mkdir -p "$ROOT/outputs"

cd "$ROOT" || exit 1

export VLA_LINGBOT_RUNTIME_BACKEND=cuda
export VLA_LINGBOT_REQUIRE_CUDA=1
export VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf
export VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf
export VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
export VLA_LINGBOT_PREDICT_TEXT_BLOCKS=24
export VLA_LINGBOT_PREDICT_MIXED=1
export VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN=1
export VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1
export VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=30
export VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q4_K
export VLA_LINGBOT_PREDICT_BLOCKS=30

if [[ ! -x "$SERVER" ]]; then
  echo "missing server binary: $SERVER" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "missing LIBERO python: $PY" >&2
  exit 1
fi
for path in "$MODEL" "$VLA_LINGBOT_TEXT_GGUF" "$VLA_LINGBOT_VAE_GGUF" "$TOKENIZER"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$SUMMARY" ]]; then
  cat > "$SUMMARY" <<'EOF'
# Overnight LingBot-VA Q4_K LIBERO Runs

| Time | Run | Task | Episodes | video/action | CFG | Status | SR | Avg inf ms | Max VRAM MiB | Output |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---|
EOF
fi

max_vram_from_log() {
  local log="$1"
  awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}' "$log" 2>/dev/null
}

field_from_summary() {
  local summary_file="$1"
  local key="$2"
  if [[ -f "$summary_file" ]]; then
    grep -m1 "^$key" "$summary_file" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '\r'
  fi
}

append_summary_row() {
  local name="$1" task_id="$2" episodes="$3" video_steps="$4" action_steps="$5" cfg="$6"
  local status="$7" out_dir="$8" vram_log="$9"
  local task_dir="$out_dir/lingbot_va/libero_90/task_$task_id"
  local summary_file="$task_dir/summary.txt"
  local sr inf vram
  sr="$(field_from_summary "$summary_file" "Success rate")"
  inf="$(field_from_summary "$summary_file" "Average inference time per step")"
  vram="$(max_vram_from_log "$vram_log")"
  [[ -n "$sr" ]] || sr="NA"
  [[ -n "$inf" ]] || inf="NA"
  [[ -n "$vram" ]] || vram="0"
  printf '| %s | %s | %s | %s | %s/%s | %s | %s | %s | %s | %s | `%s` |\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$name" "$task_id" "$episodes" \
    "$video_steps" "$action_steps" "$cfg" "$status" "$sr" "$inf" "$vram" "$out_dir" >> "$SUMMARY"
}

run_case() {
  local name="$1" task_id="$2" episodes="$3" video_steps="$4" action_steps="$5" cfg="$6" out_dir="$7"
  local server_log="$out_dir/server.log"
  local client_log="$out_dir/client.log"
  local vram_log="$out_dir/vram_mib.log"
  local status="OK"
  local server_pid=""
  local vram_pid=""

  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] starting $name task=$task_id video=$video_steps action=$action_steps cfg=$cfg" | tee "$out_dir/run.log"

  export VLA_LINGBOT_PREDICT_VIDEO_STEPS="$video_steps"
  export VLA_LINGBOT_PREDICT_ACTION_STEPS="$action_steps"
  export VLA_LINGBOT_VIDEO_GUIDANCE_SCALE="$cfg"

  (
    while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1
      sleep 5
    done
  ) > "$vram_log" &
  vram_pid=$!

  "$SERVER" --bind "$ADDR" "$MODEL" > "$server_log" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 240); do
    if grep -q "ready" "$server_log" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      status="SERVER_EXIT"
      break
    fi
    sleep 1
  done

  if [[ "$status" == "OK" ]] && ! grep -q "ready" "$server_log" 2>/dev/null; then
    status="SERVER_TIMEOUT"
  fi

  if [[ "$status" == "OK" ]]; then
    "$PY" eval/client/run_sim_client_direct.py \
      --arch lingbot_va \
      --libero-suite long \
      --task-id "$task_id" \
      --n-episodes "$episodes" \
      --max-steps 520 \
      --n-action-steps 16 \
      --recv-timeout-ms 1800000 \
      --tokenizer "$TOKENIZER" \
      --vla-addr "$ADDR" \
      --output-dir "$out_dir" \
      > "$client_log" 2>&1 || status="CLIENT_FAIL"
  fi

  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ -n "$vram_pid" ]]; then
    kill "$vram_pid" 2>/dev/null || true
    wait "$vram_pid" 2>/dev/null || true
  fi

  append_summary_row "$name" "$task_id" "$episodes" "$video_steps" "$action_steps" "$cfg" "$status" "$out_dir" "$vram_log"
  echo "[$(date '+%F %T')] finished $name status=$status" | tee -a "$out_dir/run.log"
  return 0
}

run_case "primary_A_orig_20v50a" 0 1 20 50 5 "$ROOT/outputs/overnight_lingbot_va_q4k_libero_long_task0_orig_20v50a"
run_case "primary_B_orig_20v50a" 1 1 20 50 5 "$ROOT/outputs/overnight_lingbot_va_q4k_libero_long_task1_orig_20v50a"
run_case "ablate_task0_10v25a" 0 1 10 25 5 "$ROOT/outputs/overnight_lingbot_va_q4k_libero_long_task0_ablate_10v25a"
run_case "ablate_task0_5v10a" 0 1 5 10 5 "$ROOT/outputs/overnight_lingbot_va_q4k_libero_long_task0_ablate_5v10a"
run_case "cfg1_task0_20v50a" 0 1 20 50 1 "$ROOT/outputs/overnight_lingbot_va_q4k_libero_long_task0_cfg1"

echo "All scheduled overnight LingBot-VA runs finished at $(date '+%F %T')" | tee -a "$SUMMARY"
