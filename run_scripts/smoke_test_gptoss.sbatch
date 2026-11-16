#!/bin/bash
# smoke test of the gpt-oss-20b vLLM server only (no judge) on one small GPU, then run_scripts/smoke_test_gptoss.py
# usage (from the project root):
#   VLLM_API_KEY=<key> sbatch run_scripts/smoke_test_gptoss.sbatch
#   VLLM_API_KEY=<key> sbatch --partition=<other> --gres=gpu:<type>:1 run_scripts/smoke_test_gptoss.sbatch
# NOTE gpt-oss-20b needs ~14 GB for the MXFP4 weights + KV cache -> GPU with >= 24 GB (e.g. A100, L40S, H100);
#      on GPUs older than Hopper vLLM falls back to Marlin kernels for MXFP4 (slower, fine for a smoke test)
#SBATCH --job-name=ci-impacts-smoke
#SBATCH --partition=h200_short  # pick a partition with a small GPU: sinfo -o "%P %N %G"
#SBATCH --account=swn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00         # server startup (download/compile on first run) + a few requests
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.%N.err

set -euo pipefail

 PROJECTROOT=${PROJECTROOT:-$HOME/_PROJECTS/CI-impacts-information-retrieval}
 VLLM_ENV=${VLLM_ENV:-$HOME/_PROJECTS/vllm/.venv}
 PYTHON_ENV=${PYTHON_ENV:-$PROJECTROOT/.venv}
 LOG_DIR=$(realpath -m "${LOG_DIR:-../logs}")
 mkdir -p "$LOG_DIR"

 set -a
 source "$PROJECTROOT/.env"
 set +a

 GPTOSS_PORT=${GPTOSS_PORT:-8000}
 GPTOSS_MEM=${GPTOSS_MEM:-0.85}   # gpt-oss alone on the GPU
 export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # set HF_HOME in .env to use another cache
 : "${VLLM_API_KEY:?set VLLM_API_KEY, same value as in the .env of the project}"

 source "$VLLM_ENV/bin/activate"
 export PYTHONUNBUFFERED=1
 GPTOSS_LOG="$LOG_DIR/vllm-gptoss-smoke-${SLURM_JOB_ID:-local}.log"
 echo "vLLM gpt-oss-20b on $(hostname):$GPTOSS_PORT (log: $GPTOSS_LOG)"
 nvidia-smi --query-gpu=name,memory.total --format=csv

 # same flags as in serve_vllm_both_h200.sbatch
 vllm serve openai/gpt-oss-20b \
     --served-model-name openai/gpt-oss-20b \
     --host 0.0.0.0 --port "$GPTOSS_PORT" \
     --enable-auto-tool-choice --tool-call-parser openai \
     --reasoning-parser openai_gptoss \
     --max-model-len 16384 \
     --gpu-memory-utilization "$GPTOSS_MEM" \
     --api-key "$VLLM_API_KEY" \
     >"$GPTOSS_LOG" 2>&1 &
 PID=$!
 trap 'kill $PID 2>/dev/null || true; wait 2>/dev/null || true' EXIT INT TERM

 for ((t = 0; t < 1800; t += 10)); do
     kill -0 "$PID" 2>/dev/null || { echo "ERROR: gpt-oss exited during startup, see $GPTOSS_LOG"; tail -n 30 "$GPTOSS_LOG"; exit 1; }
     curl -sf "http://localhost:$GPTOSS_PORT/health" >/dev/null && { echo "gpt-oss ready after ${t}s"; break; }
     sleep 10
 done
 grep -i "reasoning" "$GPTOSS_LOG" | head -n 3 || true   # shows which reasoning parser vLLM loaded

 deactivate
 cd "$PROJECTROOT"
 export LLM1_BASE_URL="http://localhost:$GPTOSS_PORT/v1"
 source "$PYTHON_ENV/bin/activate"
 uv run python run_scripts/smoke_test_gptoss.py   # exit code of the job = result of the smoke test
