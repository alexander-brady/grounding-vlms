#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=pmlr

SCRATCH_DIR=$SCRATCH/pmlr/

mkdir -p "$SCRATCH_DIR"

# Check if venv exists, create if not
if [ ! -d "$SCRATCH_DIR/venv" ]; then
  python3 -m venv "$SCRATCH_DIR/venv"
fi

# Activate the virtual environment
source "$SCRATCH_DIR/venv/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

srun -A pmlr -t 5 python src/run_eval.py 
    --config openai/gpt-4-1 
    --batch_size -1 
    --datasets "Sample" 