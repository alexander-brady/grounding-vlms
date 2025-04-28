#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=100G
#SBATCH --mem-per-cpu=128G
#SBATCH --time=24:00:00
#SBATCH --account=pmlr


SCRATCH_DIR=/scratch/$USER
#MODEL="openai/gpt-4-1"
MODEL="huggingface/gemma-3-4b-it"

# Clear the scratch directory
rm -rf $SCRATCH_DIR/*

# Load the necessary modules
module load cuda/11.8.0

# Check if venv exists, create if not
if [ ! -d "$SCRATCH_DIR/pmlr/.venv" ]; then
  python3 -m venv "$SCRATCH_DIR/pmlr/.venv"
fi

# Activate the virtual environment
source "$SCRATCH_DIR/pmlr/.venv/bin/activate"

echo "Starting Benchmarking job for $MODEL"
echo "Job started at $(date)"

# Upgrade pip and install requirements
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

python src/run_eval.py \
  --config $MODEL \
  --batch_size -1 \
  --datasets "Sample"
#srun -A pmlr -t 5 python src/run_eval.py --config huggingface/gemma-3-4b-it.yaml --batch_size -1 --datasets "Sample"

echo "Job completed at $(date)"

deactivate 