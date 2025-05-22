#!/bin/bash
#SBATCH --job-name=eval_hf
#SBATCH --output=logs/%u/%x_%j.out
#SBATCH --error=logs/%u/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=es_sachan
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpumem:32g

# for 12 gemma -> gpu 32 cpu 64 batch 1 worked -> 16 with batchsize 4 didnt work -> 32g batch 4 tbd 32 cpu didnt work ->32 with fast mode batch 1


models=(
    "huggingface/gemma-3-12b-it"
    "huggingface/gemma-3-27b-it"
)
MODEL=${models[$SLURM_ARRAY_TASK_ID]}

echo "$USER starting Benchmarking job for $MODEL"
echo "Job started at $(date)"

# Load the necessary modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

VENV_PATH="$SCRATCH/pmlr/$MODEL/.venv"

## Delete the venv
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
  --datasets "Sample" 
  # --datasets "FSC-147, TallyQA" \

echo "Job completed at $(date)"

deactivate 