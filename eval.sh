#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=64G
#SBATCH --time=24:00:00
#SBATCH --account=es_sachan

MODEL=${1:-"openai/gpt-4-1"}
# MODEL="huggingface/gemma-3-4b-it"

# Load the necessary modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Check if venv exists, create if not
if [ ! -d "$SCRATCH/pmlr/$MODEL/.venv" ]; then
  python3 -m venv "$SCRATCH/pmlr/$MODEL/.venv"
  echo "Virtual environment created at $SCRATCH/pmlr/$MODEL/.venv"
fi

# Activate the virtual environment
source "$SCRATCH/pmlr/$MODEL/.venv/bin/activate"

echo "Starting Benchmarking job for $MODEL"
echo "Job started at $(date)"

# Upgrade pip and install requirements
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

python src/run_eval.py \
  --config $MODEL \
  --batch_size -1 \
  --datasets "FSC-147, GeckoNum, PixMo_Count"

echo "Job completed at $(date)"

deactivate 