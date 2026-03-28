#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --job-name=jupyter-lx
#SBATCH --account=swn
#SBATCH --output=./logs/jupyter-lx-ollama.%j.%N.out 
#SBATCH --error=./logs/jupyter-lx-ollama.%j.%N.err
#SBATCH --mail-user=anna.buch@tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1  # run 1 instance of jupyter (no distributed parallelism)
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --time=03:00:00



module load $MODULES  # load all modules, 
##  TODO test if LLM geolocation also runs when only CUDA 12.1 is loaded
## NOTE: no need to export CUDA PATH and LD_LIBRARY_PATH as directly loaded via module



VENV_PATH=".venv"  
# VENV_PATH="./ollama_venv" # original env created in setup script only for ollama
OLLAMA_DIR="ollama_install"
OLLAMA_MODEL="llama3"  
NOTEBOOK_LOGFILE="./logs/jupyter-lx-run.out"
compute_node=$(hostname -f)


set -e

if 
    [ -d "${VENV_PATH}" ]; then
    echo "Virtual environment exists, activating"
    source ${VENV_PATH}"/bin/activate"
else
    echo "Run batch-ollama-setup.sh first."
    exit 1
fi


if 
   [ -f "${NOTEBOOK_LOGFILE}" ]; then 
   echo "Removing existing ${NOTEBOOK_LOGFILE}"
   rm ${NOTEBOOK_LOGFILE} 
fi


echo
echo "######################################"
echo "Configuring and starting Ollama"
echo "######################################"
echo

cd ../  # load ollama from _PROJECTS folder
export PATH=${PATH}:${PWD}/${OLLAMA_DIR}/usr/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/${OLLAMA_DIR}/usr/lib
export OLLAMA_MODELS=${PWD}/${OLLAMA_DIR}/models

# Avoid port conflicts
export OLLAMA_GAME_PORT=$((21000 + ($RANDOM % 500)))
export OLLAMA_HOST=http://127.0.0.1:${OLLAMA_GAME_PORT} 
# NOTE: make sure to have protocol scheme "http://" and that Ollama port is pointing to localhost

echo "Ollama host is ${OLLAMA_HOST}"


pushd ./CI-impacts-information-retrieval
echo "Starting ollama server"
ollama serve &
echo "Waiting for the server to initialize"
sleep 10



echo
echo "######################################"
echo "Starting jupyter on compute ${compute_node} at port ${port}"
echo "######################################"
echo

echo "Finding available port for jupyter session "

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
echo "######################################"
echo "Starting interactive session"
echo "######################################"
echo

echo "Ollama host is ${OLLAMA_HOST} and model_name ${OLLAMA_MODEL}"
source ${VENV_PATH}/bin/activate
$VENV_PATH/bin/jupyter notebook --no-browser --ip="0.0.0.0" --port=${port} --IdentityProvider.token=youshallnotpass >> ${NOTEBOOK_LOGFILE} 2>&1
# uv run python notebooks/langextract_run.py --host_port ${OLLAMA_HOST} --model_name llama3
popd


echo
echo "######################################"
echo "Done"
echo "######################################"
echo