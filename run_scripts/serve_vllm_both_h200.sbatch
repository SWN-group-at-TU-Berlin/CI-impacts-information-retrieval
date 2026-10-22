#!/bin/bash
# both vLLM servers in one job on ONE H200 NVL 141 GB, OpenAI-compatible APIs on the same node:
#   LLM 1 (extraction): openai/gpt-oss-20b              on port 8000
#   judge:              google/gemma-4-31B-it (FP8)     on port 8001
# usage:    VLLM_API_KEY=<key> sbatch scripts/serve_vllm_both_h200.sbatch
# notebook: LLM1_BASE_URL=http://<node>:8000/v1, JUDGE_BASE_URL=http://<node>:8001/v1 (on the HPC)
#           or via ssh tunnel http://localhost:8000/v1 and http://localhost:8001/v1 (local machine)
# NOTE memory split of the 141 GB (--gpu-memory-utilization = share of the total GPU memory per server):
#      gpt-oss-20b  0.20 -> ~28 GB (MXFP4 weights ~14 GB + KV cache)
#      gemma FP8    0.70 -> ~99 GB (FP8 weights ~32 GB + KV cache), ~14 GB stays free for CUDA contexts
#      sum must stay < ~0.92, both servers share the same GPU
# NOTE the servers start one after the other: vLLM checks the free GPU memory at startup,
#      parallel startups on the same GPU can both fail
# NOTE H200 (Hopper, sm_90) has native FP8 compute -> gemma FP8 checkpoint runs with FP8 kernels (no Marlin fallback)
# NOTE driver 595.45.04 supports CUDA up to 13.x -> cuda/12.8 module is fine (vLLM wheels bundle their own CUDA runtime anyway)
# NOTE needs vLLM >= 0.19.1 (Gemma 4)
#SBATCH --job-name=ci-impacts-vllm
#SBATCH --partition=TestAndBuild    # h200_short    # check which partition hosts the H200 node: sinfo -o "%P %N %G"
#SBATCH --gres=gpu:tesla:1           # one H200, both servers on it; adjust gres name to the output of sinfo
#SBATCH --cpus-per-task=2   #8          # org: 16  # 8 per server
#SBATCH --mem=10GB   ##96G
#SBATCH --time=00:30:00
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.%N.err


set -euo pipefail

module load cuda/12.8
# module load $MODULES  # load all modules
## not needed to export CUDA PATH and LD_LIBRARY_PATH as directly loaded via module

NODE=$(hostname)
PROJECTROOT=${PROJECTROOT:-$HOME/_PROJECTS/ci-impacts-vllm/}
# # NOTE: VAR=${VAR:-$xxx} inidcates a default and not fix value anymore, ie. enables to default when submitting job

set -a
source $PROJECTROOT/.env
set +a
# NOTE: -/+a"" vars defines of vars in .env export to downstream process, eg vllm

VLLM_ENV=${VLLM_ENV:-$HOME/_PROJECTS/vllm/.venv}   # separate environment with vllm installed (not the project venv)
PYTHON_ENV=${PYTHON_ENV:-$PROJECTROOT/.venv}   # project venv (jupyter)
LOG_DIR=${LOG_DIR:-"../logs"}

## model ports, memory shares, repo, maxtokens for reasoning
GPTOSS_PORT=${GPTOSS_PORT:-8000}
GEMMA_PORT=${GEMMA_PORT:-8001}
GPTOSS_MEM=${GPTOSS_MEM:-0.20}  # needed as both models share the same gpu
GEMMA_MEM=${GEMMA_MEM:-0.70}
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

# both servers are pinned to the same (single) GPU
GPU=${CUDA_VISIBLE_DEVICES:?no GPU allocated}
[[ "$GPU" != *,* ]] || echo "WARNING: more than one GPU allocated ($GPU), using only the first one"
GPU=${GPU%%,*}

GPTOSS_LOG="$LOG_DIR/vllm-gptoss-${SLURM_JOB_ID:-local}.log"
GEMMA_LOG="$LOG_DIR/vllm-gemma-${SLURM_JOB_ID:-local}.log"
echo "vLLM gpt-oss-20b on $NODE:$GPTOSS_PORT, GPU $GPU, mem share $GPTOSS_MEM (log: $GPTOSS_LOG)"
echo "vLLM $GEMMA_REPO (served as $GEMMA_SERVED_NAME) on $NODE:$GEMMA_PORT, GPU $GPU, mem share $GEMMA_MEM (log: $GEMMA_LOG)"
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
# no tool-call flags: the judge does not call tools
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
