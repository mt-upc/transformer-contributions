#!/bin/bash
#SBATCH -p veu             # Partition to submit to
#SBATCH --job-name=finetuning
#SBATCH --mem=20G      # Max CPU Memory 
#SBATCH -x veuc01,veuc05,veuc06
#SBATCH --gres=gpu:1
#SBATCH --output=/home/usuaris/veu/javier.ferrando/logs_jupyter/%j.out

set -ex

export LC_ALL=en_US.UTF-8
export PATH=~/anaconda3/bin:$PATH

export PYTHONUNBUFFERED=TRUE

export MODELS_DIR=/home/usuaris/scratch/javier.ferrando/checkpoints/


source activate alti

#mkdir -p $MODELS_DIR/multiberts
dataset=yelp
seed=0

#bert-base-uncased
# for seed in {6..10}
#     do
python finetuning/train.py --model-path distilbert-base-uncased --dataset $dataset
# done