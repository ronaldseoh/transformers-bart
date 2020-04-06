#!/bin/bash
#SBATCH --job-name=qa-to-nli
#SBATCH -p 1080ti-short
#SBATCH -N 1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bart-jupyter%j.out
#SBATCH --mem=100GB

#module load python3/3.7.3
#module load cuda101

#cd /mnt/nfs/scratch1/dhruveshpate/nli_for_qa/transformers/examples
#source .venv/bin/activate

#export PYTHONPATH=".":"${PYTHONPATH}"

jupyter-notebook --no-browser --ip=0.0.0.0
