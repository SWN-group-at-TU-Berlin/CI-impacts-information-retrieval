#!/bin/bash
# both vLLM servers + the Python workflow in ONE job on TWO A100 40 GB (one server per GPU, everything on the compute node):
#   LLM 1 (extraction): openai/gpt-oss-20b              on port 8000, GPU 0
#   judge:              google/gemma-4-31B-it (FP8)     on port 8001, GPU 1
#   workflow:           $SCRIPT (default notebooks/extract_ciimpacts.py), started once both servers are ready,
#                       talks to the servers via http://localhost:<port>/v1 (no ssh tunnel needed)
# usage (from the project root):
#   VLLM_API_KEY=<key> sbatch run_scripts/serve_vllm_both_2xa100.sbatch
#   VLLM_API_KEY=<key> GEMMA_MAX_LEN=8192 sbatch run_scripts/serve_vllm_both_2xa100.sbatch          # less KV cache needed
#   VLLM_API_KEY=<key> GEMMA_REPO=<4-bit repo> sbatch run_scripts/serve_vllm_both_2xa100.sbatch     # 4-bit judge
#   VLLM_API_KEY=<key> SCRIPT=run_scripts/smoke_test_gptoss.py sbatch run_scripts/serve_vllm_both_2xa100.sbatch
# NOTE memory per 40 GB GPU (--gpu-memory-utilization = share of the GPU memory for the server):
#      gpt-oss-20b  0.85 -> ~34 GB (MXFP4 weights ~14 GB + KV cache), plenty
#      gemma FP8    0.94 -> ~37.6 GB (FP8 weights ~32 GB) -> only a few GB KV cache left, check the
#                   "GPU KV cache size" line printed below; if gemma does not start, lower GEMMA_MAX_LEN or use a 4-bit checkpoint
# NOTE llm_judge max_tokens in extract_ciimpacts.py: prompt + max_tokens must be <= GEMMA_MAX_LEN, otherwise vLLM rejects the request
# NOTE A100 (Ampere, sm_80) has no native FP8 compute -> FP8 and MXFP4 checkpoints run weight-only (Marlin kernels), slower than on H200
# NOTE no cuda module needed: vLLM/torch wheels bundle their own CUDA runtime (a module can even break the import)
# NOTE needs vLLM >= 0.19.1 (Gemma 4)
#SBATCH --job-name=ci-impacts-vllm-a100
#SBATCH --partition=TestAndBuild  # pick a partition with 2 A100 on one node: sinfo -o "%P %N %G"; check its max time
#SBATCH --account=swn
#SBATCH --mail-user=anna.buch@tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:tesla:2     # 2 A100 on the same node; adjust gres name to the output of sinfo
#SBATCH --cpus-per-task=16     # 8 per server
#SBATCH --time=02:00:00        # server startups + the workflow
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.%N.err

set -euo pipefail

 NODE=$(hostname)
 # NOTE: VAR=${VAR:-xxx} is a default that can be overridden when submitting, e.g. SCRIPT=... sbatch ...
 PROJECTROOT=${PROJECTROOT:-$HOME/_PROJECTS/ci-impacts-vllm}
 VLLM_ENV=${VLLM_ENV:-$HOME/_PROJECTS/vllm/.venv}   # separate environment with vllm installed (not the project venv)
 PYTHON_ENV=${PYTHON_ENV:-$PROJECTROOT/.venv}       # project venv (runs the workflow)
 SCRIPT=${SCRIPT:-notebooks/extract_ciimpacts.py}   # workflow script, relative to PROJECTROOT
 # absolute log dir (resolved from the submit dir, same place as --output/--error), so a later cd does not break it
 LOG_DIR=$(realpath -m "${LOG_DIR:-../logs}")
 mkdir -p "$LOG_DIR"

 # NOTE: set -a / +a exports all vars of .env to downstream processes (vllm and the workflow)
 set -a
 source "$PROJECTROOT/.env"
 set +a

 ## model ports, memory shares, repo, context length
 GPTOSS_PORT=${GPTOSS_PORT:-8000}
 GEMMA_PORT=${GEMMA_PORT:-8001}
 GPTOSS_MEM=${GPTOSS_MEM:-0.85}  # gpt-oss alone on GPU 0
 GEMMA_MEM=${GEMMA_MEM:-0.94}    # gemma alone on GPU 1, as much as possible for the KV cache
 GEMMA_REPO=${GEMMA_REPO:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
 GEMMA_SERVED_NAME=${GEMMA_SERVED_NAME:-google/gemma-4-31B-it}   # = JUDGE_MODEL in src/settings.py, keep it when GEMMA_REPO changes
 GEMMA_MAX_LEN=${GEMMA_MAX_LEN:-16384}   # context length; lower (e.g. 8192) if the KV cache is too small, then also decrease max_tokens in llm_judge
 # extra flags to save memory on GPU 1, e.g. "--enforce-eager --max-num-seqs 4" (no CUDA graphs, fewer parallel requests)
 GEMMA_EXTRA_ARGS=${GEMMA_EXTRA_ARGS:-}

 export HF_HOME=${HF_HOME:-/beegfs/home/users/a/a-buch/_PROJECTS/huggingface_mirror}
 : "${VLLM_API_KEY:?set VLLM_API_KEY, same value as in the .env of the project}"
 [[ -f "$PROJECTROOT/$SCRIPT" ]] || { echo "ERROR: workflow script $PROJECTROOT/$SCRIPT not found"; exit 1; }


 echo
 echo "##############################################################################"
 echo "Starting vLLM servers on compute node ${NODE}"
 echo "##############################################################################"
 echo

 source "$VLLM_ENV/bin/activate"
 export PYTHONUNBUFFERED=1
 echo "vllm binary: $(command -v vllm)"

 # import check: fails fast (with a stack trace in the .err file) instead of a silent 30 min hang
 rc=0
 timeout 300 python -c "import faulthandler, sys, time; faulthandler.dump_traceback_later(240, exit=True); t=time.time(); import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, f'({time.time()-t:.0f}s)', flush=True); print('gpu:', torch.cuda.is_available(), torch.cuda.get_device_name(0), f'({time.time()-t:.0f}s)', flush=True); import vllm; print('vllm', vllm.__version__, f'({time.time()-t:.0f}s)', flush=True)" || rc=$?
 [[ $rc -eq 0 ]] || { echo "ERROR: import check failed, exit code $rc (124 = timeout)"; exit 1; }

 # pin each server to its own GPU
 IFS=, read -r GPU_GPTOSS GPU_GEMMA _ <<< "${CUDA_VISIBLE_DEVICES:?no GPU allocated}"
 : "${GPU_GEMMA:?need 2 GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}"
 nvidia-smi --query-gpu=index,name,memory.total --format=csv

 GPTOSS_LOG="$LOG_DIR/vllm-gptoss-${SLURM_JOB_ID:-local}.log"
 GEMMA_LOG="$LOG_DIR/vllm-gemma-${SLURM_JOB_ID:-local}.log"
 echo "vLLM gpt-oss-20b on $NODE:$GPTOSS_PORT, GPU $GPU_GPTOSS, mem share $GPTOSS_MEM (log: $GPTOSS_LOG)"
 echo "vLLM $GEMMA_REPO (served as $GEMMA_SERVED_NAME) on $NODE:$GEMMA_PORT, GPU $GPU_GEMMA, mem share $GEMMA_MEM, max len $GEMMA_MAX_LEN (log: $GEMMA_LOG)"

 # stop both servers when the job ends, fails or is cancelled
 PIDS=()
 cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
 trap cleanup EXIT INT TERM

 # wait until the server answers on /health (no api key needed), fail if the process died or timeout
 wait_healthy() {
     local pid=$1 port=$2 name=$3 log=$4 timeout=${5:-1800}
     for ((t = 0; t < timeout; t += 10)); do
         if ! kill -0 "$pid" 2>/dev/null; then
             echo "ERROR: $name exited during startup, last lines of $log:"; tail -n 30 "$log"; return 1
         fi
         if curl -sf "http://localhost:$port/health" >/dev/null; then
             echo "$name ready after ${t}s"
             grep -E "GPU KV cache size|Maximum concurrency" "$log" || true   # how much context fits into the KV cache
             return 0
         fi
         sleep 10
     done
     echo "ERROR: $name not ready after ${timeout}s"; return 1
 }

 # 1) gpt-oss-20b on GPU 0
 # --enable-auto-tool-choice + --tool-call-parser: needed for llm.bind_tools(..., tool_choice="any") in the graph
 CUDA_VISIBLE_DEVICES=$GPU_GPTOSS vllm serve openai/gpt-oss-20b \
     --served-model-name openai/gpt-oss-20b \
     --host 0.0.0.0 --port "$GPTOSS_PORT" \
     --enable-auto-tool-choice --tool-call-parser openai \
     --reasoning-parser openai_gptoss \
     --max-model-len 16384 \
     --gpu-memory-utilization "$GPTOSS_MEM" \
     --api-key "$VLLM_API_KEY" \
     >"$GPTOSS_LOG" 2>&1 &
 PIDS+=($!)

 # 2) gemma-4-31B (judge) on GPU 1, starts in parallel (own GPU, no competition for memory)
 # --reasoning-parser gemma4: puts the reasoning into a separate field, so the judge answer (response.content) contains only the JSON list
 # shellcheck disable=SC2086  # GEMMA_EXTRA_ARGS is split into separate flags on purpose
 CUDA_VISIBLE_DEVICES=$GPU_GEMMA vllm serve "$GEMMA_REPO" \
     --served-model-name "$GEMMA_SERVED_NAME" \
     --host 0.0.0.0 --port "$GEMMA_PORT" \
     --reasoning-parser gemma4 \
     --max-model-len "$GEMMA_MAX_LEN" \
     --gpu-memory-utilization "$GEMMA_MEM" \
     --api-key "$VLLM_API_KEY" \
     $GEMMA_EXTRA_ARGS \
     >"$GEMMA_LOG" 2>&1 &
 PIDS+=($!)

 wait_healthy "${PIDS[0]}" "$GPTOSS_PORT" gpt-oss "$GPTOSS_LOG"
 wait_healthy "${PIDS[1]}" "$GEMMA_PORT" gemma "$GEMMA_LOG"

 echo "both servers ready"
 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv


 echo
 echo "######################################"
 echo "Starting workflow: $SCRIPT"
 echo "######################################"
 echo

 # switch from the vllm venv to the project venv
 deactivate
 cd "$PROJECTROOT"

 # the workflow runs on this node -> reach the servers via localhost
 # (exported after sourcing .env, so these values win; settings.py reads them from the environment)
 export LLM1_BASE_URL="http://localhost:$GPTOSS_PORT/v1"
 export JUDGE_BASE_URL="http://localhost:$GEMMA_PORT/v1"
 echo "LLM1_BASE_URL=$LLM1_BASE_URL  JUDGE_BASE_URL=$JUDGE_BASE_URL"

 rc=0
 source "$PYTHON_ENV/bin/activate"
 uv run python "$SCRIPT" || rc=$?

 if [[ $rc -eq 0 ]]; then
     echo "Workflow finished"
 else
     echo "ERROR: workflow $SCRIPT failed with exit code $rc"
 fi
 exit $rc   # the EXIT trap stops both vLLM servers
