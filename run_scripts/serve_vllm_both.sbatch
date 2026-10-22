#!/bin/bash
# both vLLM servers in one job on one A100 80 GB, OpenAI-compatible APIs on the same node:
#   LLM 1 (extraction): openai/gpt-oss-20b              on port 8000
#   judge:              google/gemma-4-31B-it (FP8)     on port 8001
# usage:    VLLM_API_KEY=<key> sbatch scripts/serve_vllm_both.sbatch
# notebook: LLM1_BASE_URL=http://<node>:8000/v1, JUDGE_BASE_URL=http://<node>:8001/v1 (on the HPC)
#           or via ssh tunnel http://localhost:8000/v1 and http://localhost:8001/v1 (local machine)
# NOTE memory split of the 80 GB (--gpu-memory-utilization = share of the total GPU memory per server):
#      gpt-oss-20b  0.25 -> ~20 GB (MXFP4 weights ~14 GB + KV cache)
#      gemma FP8    0.68 -> ~54 GB (FP8 weights ~32 GB + KV cache), rest stays free for CUDA contexts
# NOTE the servers start one after the other: vLLM checks the free GPU memory at startup,
#      parallel startups can both fail
# NOTE A100 has no native FP8 compute -> vLLM runs the FP8 checkpoint weight-only (Marlin), slower than on H100/H200
# NOTE needs vLLM >= 0.19.1 (Gemma 4); Tesla GPUs (V100/P100) are not supported by current vLLM -> pinned to a100
#SBATCH --job-name=ci-impacts-vllm
#SBATCH --partition=TestAndBuild
#SBATCH --gres=gpu:tesla:2  # a100   gpt-oss gpu0, gemma gp1
#SBATCH --cpus-per-task=1   # org: 16           # 8 per server
# #SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=../logs/debug-%j.out
#SBATCH --error=../logs/debug-%j.%N.err


set -euo pipefail

module load cuda/12.8
# module load $MODULES  # load all modules
## not needed to export CUDA PATH and LD_LIBRARY_PATH as directly loaded via module

NODE=$(hostname)
PROJECTROOT=${PROJECTROOT:-$HOME/_PROJECTS/ci-impacts-vllm/}
# # NOTE: VAR=${VAR:-$xxx} inidcates a default and not fix value anymore, ie. enables to default when submitting job
# # NOTE: VAR=${VAR:-$xxx} inidcates a default and not fix value anymore, ie. enables to default when submitting job

set -a
source $PROJECTROOT/.env
set +a
# NOTE: -/+a"" vars defines of vars in .env export to downstream process, eg vllm

VLLM_ENV=${VLLM_ENV:-$HOME/_PROJECTS/vllm}   # separate environment with vllm installed (not the project venv)
PYTHON_ENV=${PYTHON_ENV:-$PROJECTROOT/.venv}   # separate environment with vllm installed (not the project venv)
LOG_DIR=${LOG_DIR:-"../logs"}

## model ports, memory shares, repo, maxtokens for reasoning
GPTOSS_PORT=${GPTOSS_PORT:-8000}
GEMMA_PORT=${GEMMA_PORT:-8001}
# GPTOSS_MEM=${GPTOSS_MEM:-0.25}  # only needed when both model on same gpu
# GEMMA_MEM=${GEMMA_MEM:-0.68}
GEMMA_REPO=${GEMMA_REPO:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
GEMMA_SERVED_NAME=${GEMMA_SERVED_NAME:-google/gemma-4-31B-it}   # = JUDGE_MODEL in src/settings.py
GEMMA_MAX_LEN=${GEMMA_MAX_LEN:-16384}   # context length; lower if kv cache too small, then also need to decrease max_tokens in llm_judge

export HF_HOME=${HF_HOME:-/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror}
: "${VLLM_API_KEY:?set VLLM_API_KEY, same value as in the .env of the project}"



echo
echo "##############################################################################"
echo "Starting vLLM server on compute node ${NODE}"
echo "##############################################################################"
echo

source "$VLLM_ENV/bin/activate"

IFS=, read -r GPU_GPTOSS GPU_GEMMA <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}" # Pin each server to its own GPU
: "${GPU_GEMMA:?need 2 GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}" 

NODE=$(hostname)
GPTOSS_LOG="$LOG_DIR/vllm-gptoss-${SLURM_JOB_ID:-local}.log"
GEMMA_LOG="$LOG_DIR/vllm-gemma-${SLURM_JOB_ID:-local}.log"
echo "vLLM gpt-oss-20b on $NODE:$GPTOSS_PORT (log: $GPTOSS_LOG)"
echo "vLLM $GEMMA_REPO (served as $GEMMA_SERVED_NAME) on $NODE:$GEMMA_PORT (log: $GEMMA_LOG)"
echo "local machine: ssh -N -L $GPTOSS_PORT:$NODE:$GPTOSS_PORT -L $GEMMA_PORT:$NODE:$GEMMA_PORT gateway.hpc.tu-berlin.de" # <user>@<hpc-login>"
# NOTE ":" reports success/fail based on later check; ":?" check if Var is empty

# stop both servers when the job ends or one of them crashes
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
CUDA_VISIBLE_DEVICES=$GPU_GPTOSS vllm serve openai/gpt-oss-20b \
    --served-model-name openai/gpt-oss-20b \
    --host 0.0.0.0 --port "$GPTOSS_PORT" \
    --enable-auto-tool-choice --tool-call-parser openai \
    --max-model-len 16384 \
    --gpu-memory-utilization "$GPTOSS_MEM" \
    --api-key "$VLLM_API_KEY" \
    >"$GPTOSS_LOG" 2>&1 &
PIDS+=($!)
wait_healthy "${PIDS[0]}" "$GPTOSS_PORT" gpt-oss

# 2) gemma-4-31B FP8 (judge), only after gpt-oss holds its memory
# --reasoning-parser gemma4: puts the reasoning into a separate field, so the judge answer (response.content) contains only the JSON list
# no tool-call flags: the judge does not call tools
CUDA_VISIBLE_DEVICES=$GPU_GEMMA vllm serve "$GEMMA_REPO" \
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
echo "##############################################################################"
echo "Starting Jupyer Notebook Server on same compute node"
echo "##############################################################################"
echo

NOTEBOOK_LOGFILE=${LOG_DIR}/jupyterlog_gpu.out

if 
   [ -f "${NOTEBOOK_LOGFILE}" ]; then 
   echo "Removing existing ${NOTEBOOK_LOGFILE}"
   rm ${NOTEBOOK_LOGFILE} 
fi


check_port() {
   nc -z localhost $1
   return $(( ! $? ))
}

# Find an available port
port=8890
while ! check_port $port; do
   port=$((port + 1))
done

echo
echo "##############################################################################"
echo "Starting jupyter server on node at port ${port}"
echo "##############################################################################"
echo

$PYTHON_ENV/bin/jupyter notebook --no-browser --ip="0.0.0.0" --port=${port} --IdentityProvider.token=youshallnotpass >> ${NOTEBOOK_LOGFILE} 2>&1



# job runs until one server exits (crash) or the time limit is reached, then the trap stops the other one
wait -n "${PIDS[@]}"
echo "ERROR: one server exited, stopping the job"
exit 1

