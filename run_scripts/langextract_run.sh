#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --job-name=ollama-run
#SBATCH --account=swn
#SBATCH --output=./logs/ollama-run.%j.%N.out 
#SBATCH --error=./logs/ollama-run.%j.%N.err
#SBATCH --mail-user=anna.buch@tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5   # test if performance change when using less cpus
#SBATCH --gpus=3   #  test if performance change between 1 or 3 (max) gpus per node
#SBATCH --mem=10G
#SBATCH --time=15:00:00

# performance check
# ~ 6h processing time: 2 nodes,  5 cpus per task, 4 gpus, 

module purge
# module load nvidia/cuda/12.2 # check if  performance changes when CUDA module is not loaded


# VENV_PATH="./ollama_venv" # original env created in setup script only for ollama
VENV_PATH="./CI-impacts-information-retrieval/.venv"  # uv
OLLAMA_DIR="ollama_install"


set -e
cd ../  # load ollama from _PROJECTS folder


if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists, activating"
    source "${VENV_PATH}"/bin/activate
else
    echo "Run batch-ollama-setup.sh first."
    exit 1
fi


echo
echo "######################################"
echo "Configuring Ollama"
echo "######################################"
echo
export PATH=${PATH}:${PWD}/${OLLAMA_DIR}/usr/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/${OLLAMA_DIR}/usr/lib
export OLLAMA_MODELS=${PWD}/${OLLAMA_DIR}/models

# Avoid port conflicts
export OLLAMA_GAME_PORT=$((21000 + ($RANDOM % 500)))
export OLLAMA_HOST=http://127.0.0.1:${OLLAMA_GAME_PORT} 
# NOTE: make sure to have protocol scheme "http://" and that Ollama port is pointing to localhost

echo "Ollama host is ${OLLAMA_HOST}"

echo
echo "######################################"
echo "Starting LangExtract"
echo "######################################"
echo

pushd ./CI-impacts-information-retrieval
echo "Starting ollama server"
ollama serve &
echo "Waiting for the server to inialise"
sleep 10


echo "Starting script"
source .venv/bin/activate
uv run python notebooks/langextract_run.py --host_port ${OLLAMA_HOST} --model_name llama3
popd


echo
echo "######################################"
echo "Done"
echo "######################################"
echo

echo "Stopping ollama"
pkill ollama

echo "Deactivating virtual environment"
deactivate
popd
