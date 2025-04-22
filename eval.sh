#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G


source ~/.venv/bin/activate

srun python src/run_eval.py \
  --config openai/gpt-4.1 \ 
  --batch_size -1 \
  --output_dir results/benchmark \