#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=gpu_short
#SBATCH --job-name=llm-agents
#SBATCH --account=swn
#SBATCH --output=../logs/llm-agents.%j.%N.out 
#SBATCH --error=../logs/llm-agents.%j.%N.err
#SBATCH --mail-user=anna.buch@tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
# # SBATCH --mem=5G
#SBATCH --time=03:00:00


module load cuda/12.8

# TODO make script names as var

echo
echo "######################################"
echo "Starting script: llm_geolocations"
echo "######################################"
echo

source .venv/bin/activate
uv run python notebooks/llm_geolocations.py || exit 1 # return general failure  and stop process if python script fails

echo
echo "######################################"
echo "Starting script: llm_evaluation"
echo "######################################"
echo

uv run python notebooks/llm_evaluation.py  # or test sequencial run with file1.py && file2.py

echo "Workflow finished"