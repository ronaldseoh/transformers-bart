#!/bin/bash
#SBATCH --job-name=qa-to-nli
#SBATCH -p m40-short
#SBATCH -N 1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qa_to_nli_bart_%j.out
#SBATCH --mem=100GB

#module load python3/3.7.3
#module load cuda101

#cd /mnt/nfs/scratch1/dhruveshpate/nli_for_qa/transformers/examples
#source .venv/bin/activate

#export PYTHONPATH=".":"${PYTHONPATH}"

python summarization/bart/run_bart_sum.py \
--data_dir=.data/qa_to_hypothesis \
--model_type=bart \
--model_name_or_path=bart-large \
--do_lower_case \
--learning_rate=1e-5 \
--train_batch_size=24 \
--eval_batch_size=24 \
--output_dir=.models/bart \
--do_predict \
--max_seq_length 45 \
--n_gpu 1
