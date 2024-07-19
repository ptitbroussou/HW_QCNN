#!/bin/bash

#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --constraint=amd
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=80G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=1500
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

python QCNN_3D.py