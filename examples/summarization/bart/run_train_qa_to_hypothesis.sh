# Add parent directory to python path to access transformer_base.py

export PYTHONPATH="../../../src":"../../":"${PYTHONPATH}"

python run_bart_sum.py \
--data_dir=../../.data/qa_to_hypothesis_test \
--model_type=bart \
--model_name_or_path=bart-large \
--do_lower_case \
--learning_rate=3e-5 \
--train_batch_size=16 \
--eval_batch_size=16 \
--output_dir=.models/bart_test \
--do_train \
--max_seq_length 40 \
--n_gpu 1
