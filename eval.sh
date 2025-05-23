#!/bin/bash
#SBATCH --job-name=eval_hf
#SBATCH --output=logs/%u/%x_%j.out
#SBATCH --error=logs/%u/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=100G
#SBATCH --mem-per-cpu=100G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=es_sachan
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpumem:64g

# for 12 gemma -> gpu 32 cpu 64 batch 1 worked -> 16 with batchsize 4 didnt work -> 32g batch 4 tbd 32 cpu didnt work ->32 with fast mode batch 1

# MODEL=${1:-"huggingface/Qwen2-5-VL-3B-Instruct"}
# MODEL=${1:-"huggingface/Qwen2-5-VL-7B-Instruct"}
# MODEL=${1:-"huggingface/Qwen2-5-VL-72B-Instruct"}

# MODEL=${1:-"huggingface/gemma-3-4b-it"}
# MODEL=${1:-"huggingface/gemma-3-12b-it"}
MODEL=${1:-"huggingface/gemma-3-27b-it"}

# MODEL=${1:-"huggingface/Llama-4-Scout-17B-16E-Instruct"}

echo "$USER starting Benchmarking job for $MODEL"
echo "Job started at $(date)"

# Load the necessary modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

VENV_PATH="$SCRATCH/pmlr/$MODEL/.venv"

# # Delete the venv
if [ -d "$VENV_PATH" ]; then
  rm -rf "$VENV_PATH"
  echo "Cleared existing virtual environment at $VENV_PATH at $(date)"
fi

# Check if venv exists, create if not
if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
  echo "Virtual environment created at $VENV_PATH at $(date)"
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Model paths
export HF_HOME="$SCRATCH/pmlr/$MODEL/cache"

echo "Beginning evaluation for $MODEL at $(date)"

python src/run_eval.py \
  --config $MODEL \
  --datasets "GeckoNum"
  # --datasets "TallyQA"
  # --datasets "GeckoNum"
  #--datasets "FSC-147"
  # --datasets "FSC-147, TallyQA, GeckoNum"

echo "Job completed at $(date)"

deactivate 