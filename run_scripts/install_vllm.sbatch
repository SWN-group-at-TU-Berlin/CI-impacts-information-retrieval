#!/bin/bash
# installs vLLM into a separate venv on the cluster as a batch job (no interactive srun session needed)
# and checks afterwards that torch sees the GPU and vllm can be imported
#   target venv: $VLLM_ENV (default $HOME/_PROJECTS/vllm/.venv = default VLLM_ENV of the serve_vllm_*.sbatch scripts)
# usage (from the project root):
#   sbatch run_scripts/install_vllm.sbatch
#   VLLM_ENV=/work/<user>/vllm/.venv sbatch run_scripts/install_vllm.sbatch     # other target folder (e.g. if $HOME quota is small)
#   VLLM_VERSION=0.20.0 sbatch run_scripts/install_vllm.sbatch                 # other vLLM version
#   REINSTALL=1 sbatch run_scripts/install_vllm.sbatch                         # delete and recreate an existing venv
# NOTE needs vLLM >= 0.19.1 (Gemma 4), the wheels bundle torch + CUDA runtime -> no cuda module needed
# NOTE runs on a GPU node on purpose: uv's --torch-backend=auto picks the torch CUDA build that fits the
#      node's NVIDIA driver, and the final import check can test the GPU. Use the same GPU type as for serving.
# NOTE the compute node needs internet access (PyPI, download.pytorch.org). If it has none, run the commands
#      of the "install" section below on the login node instead (without the GPU check)
#SBATCH --job-name=install-vllm
#SBATCH --partition=h200_short   # any partition with a GPU of the type used for serving: sinfo -o "%P %N %G"
#SBATCH --account=swn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.%N.err

set -euo pipefail

VLLM_VERSION=${VLLM_VERSION:-0.19.1}
VLLM_ENV=${VLLM_ENV:-$HOME/_PROJECTS/vllm/.venv}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
REINSTALL=${REINSTALL:-0}
# uv cache: put it next to the venv if $HOME is small, e.g. UV_CACHE_DIR=/work/<user>/.cache/uv
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}
export UV_LINK_MODE=copy   # cache and venv may be on different file systems (no hardlinks across them)

echo "node: $(hostname), job: ${SLURM_JOB_ID:-local}"
echo "vLLM $VLLM_VERSION -> $VLLM_ENV (python $PYTHON_VERSION, uv cache $UV_CACHE_DIR)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "WARNING: nvidia-smi not available"

# uv is usually installed to ~/.local/bin, which is not always on PATH in batch jobs
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not found, install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo "uv: $(uv --version)"


## install
if [[ -d "$VLLM_ENV" ]]; then
    if [[ "$REINSTALL" == 1 ]]; then
        echo "removing existing venv $VLLM_ENV (REINSTALL=1)"
        rm -rf "$VLLM_ENV"
    else
        echo "venv $VLLM_ENV exists, installing into it (set REINSTALL=1 for a clean venv)"
    fi
fi
mkdir -p "$(dirname "$VLLM_ENV")"
[[ -d "$VLLM_ENV" ]] || uv venv --python "$PYTHON_VERSION" --seed "$VLLM_ENV"

# --torch-backend=auto: torch wheel matching the installed CUDA driver (see vLLM installation guide)
uv pip install --python "$VLLM_ENV/bin/python" "vllm==$VLLM_VERSION" --torch-backend=auto


## check: same import check as in the serve scripts
source "$VLLM_ENV/bin/activate"
echo "vllm binary: $(command -v vllm)"
rc=0
timeout 300 python -c "import faulthandler, time; faulthandler.dump_traceback_later(240, exit=True); t=time.time(); import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, f'({time.time()-t:.0f}s)', flush=True); print('gpu:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', flush=True); import vllm; print('vllm', vllm.__version__, f'({time.time()-t:.0f}s)', flush=True)" || rc=$?
[[ $rc -eq 0 ]] || { echo "ERROR: import check failed, exit code $rc (124 = timeout)"; exit 1; }

echo "done. Use it in the serve scripts with VLLM_ENV=$VLLM_ENV (default if unchanged)"
