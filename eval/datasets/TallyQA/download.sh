#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=download
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00

cd /users/teilers/scratch/grounding-vlms/eval/datasets/TallyQA/images

wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip images.zip

wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images2.zip

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip