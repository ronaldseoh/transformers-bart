#!/bin/bash
#SBATCH --job-name=bart-qa-to-nli
#SBATCH -p 1080ti-long
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bart-prediction-%j.out
#SBATCH --mem=100GB

#module load python3/3.7.3
#module load cuda101

#cd /mnt/nfs/scratch1/dhruveshpate/nli_for_qa/transformers/examples
#source .venv/bin/activate

#export PYTHONPATH=".":"${PYTHONPATH}"

python summarization/bart/BART_RACE_qa_to_nli.py --data_folder .data/RACE/set3 --model_checkpoint ".models/bart2/epoch=1_v0.ckpt" --set train --device 0
