#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=TestAndBuild
#SBATCH --job-name=ollama-run
#SBATCH --account=swn
#SBATCH --output=./logs/ollama-run.%j.%N.out 
#SBATCH --error=./logs/ollama-run.%j.%N.err
#SBATCH --mail-user=anna.buch@tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes 1
#SBATCH --cpus-per-task=2 # 5
### # --gpus=1
#SBATCH --mem=3G #5G # 32768
#SBATCH --time=00:10:00 # 01:00:00


module purge

# VENV_PATH="./ollama_venv" # original env created in setup script only for ollama
VENV_PATH="./CI-impacts-information-retrieval/.venv"  # uv
OLLAMA_DIR="ollama_install"


set -e
cd ../  # load ollama from _PROJECTS folder


if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists, activating"
    source "${VENV_PATH}"/bin/activate
else
    echo "Please sbatch the setup script before running this."
    exit 1
fi


echo
echo "######################################"
echo "Configuring ollama"
echo "######################################"
echo
export PATH=${PATH}:${PWD}/${OLLAMA_DIR}/usr/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/${OLLAMA_DIR}/usr/lib
export OLLAMA_MODELS=${PWD}/${OLLAMA_DIR}/models

# Avoid port conflicts
export OLLAMA_GAME_PORT=$((21000 + ($RANDOM % 500)))
export OLLAMA_HOST=127.0.0.1:${OLLAMA_GAME_PORT}


echo "Ollama host is ${OLLAMA_HOST}"

ollama_port="127.0.0.1:${OLLAMA_GAME_PORT}"  # arg for .py, NOTE: need to be string

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
echo $ollama_port
# when ollama_port is int/str: langextract_run.py: error: unrecognized arguments: --ollama_port=127.0.0.1:21170 --model llama3
# uv run python notebooks/langextract_run.py "--ollama_port=${ollama_port}" "--host_port ${OLLAMA_HOST}" "--model=llama3"

uv run python notebooks/langextract_run.py "--host_port ${OLLAMA_HOST}" "--model_name llama3"

# uv run python ./notebooks/langextract_run.py "--host_port ${OLLAMA_HOST}" "--model llama3" # langextract_run.py: error: unrecognized arguments: --model llama3
# uv run python ./notebooks/langextract_run.py "${ollama_port}" "--host_port ${OLLAMA_HOST}" "--model llama3"
# python3 notebooks/langextract_run.py "--host_port ${OLLAMA_HOST}" "--model llama3" # unrecognized arguments: --model llama3

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
