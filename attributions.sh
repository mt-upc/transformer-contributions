#!/bin/bash
#SBATCH -p veu             # Partition to submit to
#SBATCH --job-name=sva_bert
#SBATCH --mem=40G      # Max CPU Memory 
#SBATCH -x veuc01,veuc05,veuc06
#SBATCH --gres=gpu:1
#SBATCH --output=/home/usuaris/veu/javier.ferrando/logs_jupyter/%j.out
CUDA_VISIBLE_DEVICES=0
set -ex

export LC_ALL=en_US.UTF-8
export PATH=~/anaconda3/bin:$PATH

export PYTHONUNBUFFERED=TRUE


source activate attn_flow

dataset=sst2
num_samples=1000
bert_seed=0
model_name=multiberts-seed_$bert_seed
model_name=bert

python attributions.py  --model $model_name \
                        --dataset $dataset \
                        --samples $num_samples
for faith_metric in comp suff; do
    if [ $dataset = imdb ] || [  $dataset = yelp ]
    then
        max_skip=60
    else
        max_skip=30
    fi
    for i in {1..2}; do
    if [[ "$i" == '1' ]]
    then
        echo $max_skip
        python aupc.py  --model $model_name \
                    --dataset $dataset \
                    --samples $num_samples \
                    --fidelity-type $faith_metric \
                    --bins \
                    --max-skip $max_skip
    else
        python aupc.py  --model $model_name \
                    --dataset $dataset \
                    --samples $num_samples \
                    --fidelity-type $faith_metric \
                    --max-skip $max_skip
    fi
    done
done