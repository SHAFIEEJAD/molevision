#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64000

module load python/3.11.2
source ~/venvs/molevision/bin/activate

python /work/ws-tmp/g062484-melo/images/preprocess_and_save_parallel.py

