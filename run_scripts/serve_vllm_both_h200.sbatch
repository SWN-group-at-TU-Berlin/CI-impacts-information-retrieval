#!/bin/bash
# both vLLM servers + the Python workflow in ONE job on ONE H200 NVL 141 GB (everything runs on the compute node):
#   LLM 1 (extraction): openai/gpt-oss-20b              on port 8000
#   judge:              google/gemma-4-31B-it (FP8)     on port 8001
#   workflow:           $SCRIPT (default notebooks/langgraph.py), started once both servers are ready,
#                       talks to the servers via http://localhost:<port>/v1 (no ssh tunnel needed)
# usage (from the project root):
#   VLLM_API_KEY=<key> sbatch run_scripts/serve_vllm_both_h200.sbatch
#   VLLM_API_KEY=<key> SCRIPT=src/other_script.py sbatch run_scripts/serve_vllm_both_h200.sbatch
# NOTE memory split of the 141 GB (--gpu-memory-utilization = share of the total GPU memory per server):
#      gpt-oss-20b  0.20 -> ~28 GB (MXFP4 weights ~14 GB + KV cache)
#      gemma FP8    0.70 -> ~99 GB (FP8 weights ~32 GB + KV cache), ~14 GB stays free for CUDA contexts
#      sum must stay < ~0.92, both servers share the same GPU
# NOTE the servers start one after the other: vLLM checks the free GPU memory at startup,
#      parallel startups on the same GPU can both fail
# NOTE H200 (Hopper, sm_90) has native FP8 compute -> gemma FP8 checkpoint runs with FP8 kernels (no Marlin fallback)
# NOTE no cuda module needed: vLLM/torch wheels bundle their own CUDA runtime (a module can even break the import)
# NOTE needs vLLM >= 0.19.1 (Gemma 4)
#SBATCH --job-name=ci-impacts-vllm
#SBATCH --partition=h200_short  #TestAndBuild  # check which partition hosts the H200 node: sinfo -o "%P %N %G"
#SBATCH --account=swn
#SBATCH --mail-user=anna.buch@tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:h200:1      # one H200, both servers on it; adjust gres name to the output of sinfo
#SBATCH --cpus-per-task=16     # 8 per server
# #SBATCH --mem=96G
#SBATCH --time=02:00:00        # server startups (up to 2 x 30 min worst case) + the workflow; check the partition's max time
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.%N.err

set -euo pipefail

 NODE=$(hostname)
 # NOTE: VAR=${VAR:-xxx} is a default that can be overridden when submitting, e.g. SCRIPT=... sbatch ...
 PROJECTROOT=${PROJECTROOT:-$HOME/_PROJECTS/ci-impacts-vllm}
 VLLM_ENV=${VLLM_ENV:-$HOME/_PROJECTS/vllm/.venv}   # separate environment with vllm installed (not the project venv)
 PYTHON_ENV=${PYTHON_ENV:-$PROJECTROOT/.venv}       # project venv (runs the workflow)
 SCRIPT=${SCRIPT:-notebooks/langgraph.py}           # workflow script, relative to PROJECTROOT
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
 GPTOSS_MEM=${GPTOSS_MEM:-0.20}  # needed as both models share the same gpu
 GEMMA_MEM=${GEMMA_MEM:-0.70}
 GEMMA_REPO=${GEMMA_REPO:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
 GEMMA_SERVED_NAME=${GEMMA_SERVED_NAME:-google/gemma-4-31B-it}   # = JUDGE_MODEL in src/settings.py
 GEMMA_MAX_LEN=${GEMMA_MAX_LEN:-16384}   # context length; lower if kv cache too small, then also decrease max_tokens in llm_judge

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
 export VLLM_LOGGING_LEVEL=DEBUG   # set back to INFO once startup works
 echo "vllm binary: $(command -v vllm)"

 # import check: fails fast (with a stack trace in the .err file) instead of a silent 30 min hang
 rc=0
 timeout 300 python -c "import faulthandler, sys, time; faulthandler.dump_traceback_later(240, exit=True); t=time.time(); import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, f'({time.time()-t:.0f}s)', flush=True); print('gpu:', torch.cuda.is_available(), torch.cuda.get_device_name(0), f'({time.time()-t:.0f}s)', flush=True); import vllm; print('vllm', vllm.__version__, f'({time.time()-t:.0f}s)', flush=True)" || rc=$?
 [[ $rc -eq 0 ]] || { echo "ERROR: import check failed, exit code $rc (124 = timeout)"; exit 1; }

 # both servers are pinned to the same (single) GPU
 GPU=${CUDA_VISIBLE_DEVICES:?no GPU allocated}
 [[ "$GPU" != *,* ]] || echo "WARNING: more than one GPU allocated ($GPU), using only the first one"
 GPU=${GPU%%,*}

 GPTOSS_LOG="$LOG_DIR/vllm-gptoss-${SLURM_JOB_ID:-local}.log"
 GEMMA_LOG="$LOG_DIR/vllm-gemma-${SLURM_JOB_ID:-local}.log"
 echo "vLLM gpt-oss-20b on $NODE:$GPTOSS_PORT, GPU $GPU, mem share $GPTOSS_MEM (log: $GPTOSS_LOG)"
 echo "vLLM $GEMMA_REPO (served as $GEMMA_SERVED_NAME) on $NODE:$GEMMA_PORT, GPU $GPU, mem share $GEMMA_MEM (log: $GEMMA_LOG)"

 # stop both servers when the job ends, fails or is cancelled
 PIDS=()
 cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
 trap cleanup EXIT INT TERM

 # wait until the server answers on /health (no api key needed), fail if the process died or timeout
 wait_healthy() {
     local pid=$1 port=$2 name=$3 timeout=${4:-1800}
     for ((t = 0; t < timeout; t += 10)); do
         if ! kill -0 "$pid" 2>/dev/null; then
             echo "ERROR: $name exited during startup, see its log"; return 1
         fi
         if curl -sf "http://localhost:$port/health" >/dev/null; then
             echo "$name ready after ${t}s"; return 0
         fi
         sleep 10
     done
     echo "ERROR: $name not ready after ${timeout}s"; return 1
 }

 # 1) gpt-oss-20b (smaller, starts first)
 # --enable-auto-tool-choice + --tool-call-parser: needed for llm.bind_tools(..., tool_choice="any") in the graph
 CUDA_VISIBLE_DEVICES=$GPU vllm serve openai/gpt-oss-20b \
     --served-model-name openai/gpt-oss-20b \
     --host 0.0.0.0 --port "$GPTOSS_PORT" \
     --enable-auto-tool-choice --tool-call-parser openai \
     --max-model-len 16384 \
     --gpu-memory-utilization "$GPTOSS_MEM" \
     --api-key "$VLLM_API_KEY" \
     >"$GPTOSS_LOG" 2>&1 &
 PIDS+=($!)
 wait_healthy "${PIDS[0]}" "$GPTOSS_PORT" gpt-oss

 # 2) gemma-4-31B FP8 (judge), only after gpt-oss holds its memory on the shared GPU
 # --reasoning-parser gemma4: puts the reasoning into a separate field, so the judge answer (response.content) contains only the JSON list
 CUDA_VISIBLE_DEVICES=$GPU vllm serve "$GEMMA_REPO" \
     --served-model-name "$GEMMA_SERVED_NAME" \
     --host 0.0.0.0 --port "$GEMMA_PORT" \
     --reasoning-parser gemma4 \
     --max-model-len "$GEMMA_MAX_LEN" \
     --gpu-memory-utilization "$GEMMA_MEM" \
     --api-key "$VLLM_API_KEY" \
     >"$GEMMA_LOG" 2>&1 &
 PIDS+=($!)
 wait_healthy "${PIDS[1]}" "$GEMMA_PORT" gemma

 echo "both servers ready"
 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv


 echo
 echo "######################################"
 echo "Starting workflow: $SCRIPT"
 echo "######################################"
 echo

 # switch from the vllm venv to the project venv
 deactivate
 cd "$PROJECTROOT"

 # the workflow runs on this node -> reach the servers via localhost
 # (exported after sourcing .env, so these values win; settings.py must read them from the environment)
 export LLM1_BASE_URL="http://localhost:$GPTOSS_PORT/v1"
 export JUDGE_BASE_URL="http://localhost:$GEMMA_PORT/v1"
 echo "LLM1_BASE_URL=$LLM1_BASE_URL  JUDGE_BASE_URL=$JUDGE_BASE_URL"

 # export LLM1_BASE_URL="http://$NODE:$GPTOSS_PORT/v1"
 # export JUDGE_BASE_URL="http://$NODE:$GEMMA_PORT/v1"
 rc=0
 # "$PYTHON_ENV/bin/python" "$SCRIPT" || rc=$?   # alternative: uv run --no-sync python "$SCRIPT"
source $PYTHON_ENV/bin/activate
uv run python ${SCRIPT}  || rc=$?


 if [[ $rc -eq 0 ]]; then
     echo "Workflow finished"
 else
     echo "ERROR: workflow $SCRIPT failed with exit code $rc"
 fi
 exit $rc   # the EXIT trap stops both vLLM servers

