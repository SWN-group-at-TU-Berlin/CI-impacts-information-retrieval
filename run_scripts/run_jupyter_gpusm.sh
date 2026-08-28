#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --job-name=jupyter-interactive
#SBATCH --account=swn
#SBATCH --mail-user=anna.buch@tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1  # run 1 instance of jupyter (here equivalent to job - each task inherits the parameters specified for the batch script)
#SBATCH --nodes=1
#SBATCH --gres=gpu:2 # dedicated gpus -> eg. gpu:tesla:2, gpu:a100:1
#SBATCH --cpus-per-task=1 # no. threads (enabling multi-threading/MP), TODO decrease to 1 for serial code
# # SBATCH --mem=10G   # not needed as nobody else is likely using any of the remaining core on the 2 GPUs
#SBATCH --time=02:00:00



module load cuda/12.8
# module load $MODULES  # load all modules, TODO test if LLM geolocation lso runs when only CUDA 12.1 is loaded
## not needed to export CUDA PATH and LD_LIBRARY_PATH as directly loaded via module


PYTHONPATH=./.venv
NOTEBOOK_LOGFILE=../logs/jupyterlog_gpu.out
compute_node=$(hostname -f)


if 
   [ -f "${NOTEBOOK_LOGFILE}" ]; then 
   echo "Removing existing ${NOTEBOOK_LOGFILE}"
   rm ${NOTEBOOK_LOGFILE} 
fi


echo
echo "######################################"
echo "Finding available port"
echo "######################################"
echo

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
echo "Starting jupyter server on compute node ${compute_node} at port ${port}"
echo "######################################"
echo

$PYTHONPATH/bin/jupyter notebook --no-browser --ip="0.0.0.0" --port=${port} --IdentityProvider.token=youshallnotpass >> ${NOTEBOOK_LOGFILE} 2>&1

