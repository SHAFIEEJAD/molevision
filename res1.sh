#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64000
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load python

source ~/venvs/molevision/bin/activate

input_file="/work/ws-tmp/g062484-melo/images/train_resnet_melanoma.py"

python /work/ws-tmp/g062484-melo/images/train_resnet_melanoma.py "$input_file"

