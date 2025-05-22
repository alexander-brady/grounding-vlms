#!/bin/bash
#SBATCH --job-name=eval_api
#SBATCH --output=logs/%u/%x_%j.out
#SBATCH --error=logs/%u/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --account=es_sachan
#SBATCH --mail-type=END,FAIL

# for 12 gemma -> gpu 32 cpu 64 batch 1 worked -> 16 with batchsize 4 didnt work -> 32g batch 4 tbd 32 cpu didnt work ->32 with fast mode batch 1

# MODEL=${1:-"openai/gpt-4-1"}
# MODEL=${1:-"openai/o4-mini"}

MODEL=${1:-"google/gemini-2.5-flash-preview-04-17"}
# MODEL=${1:-"google/gemini-2.5-pro-preview-05-06"}

# MODEL=${1:-"xai/grok-2-vision"}

# MODEL=${1:-"anthropic/claude-3.5-haiku"}


# Load the necessary modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

## Delete the venv
# if [ -d "$SCRATCH/pmlr/$MODEL/.venv" ]; then
#   rm -rf "$SCRATCH/pmlr/$MODEL/.venv"
#   echo "Cleared existing virtual environment at $SCRATCH/pmlr/$MODEL/.venv"
# fi

# Check if venv exists, create if not
if [ ! -d "$SCRATCH/pmlr/$MODEL/.venv" ]; then
  python3 -m venv "$SCRATCH/pmlr/$MODEL/.venv"
  echo "Virtual environment created at $SCRATCH/pmlr/$MODEL/.venv"
fi

# Activate the virtual environment
source "$SCRATCH/pmlr/$MODEL/.venv/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Model paths
export HF_HOME="$SCRATCH/pmlr/$MODEL/cache"

echo "$USER starting Benchmarking job for $MODEL"
echo "Job started at $(date)"

python src/run_eval.py \
  --config $MODEL \
  --datasets "Missing"

echo "Job completed at $(date)"

deactivate 