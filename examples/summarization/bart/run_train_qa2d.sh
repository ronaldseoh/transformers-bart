export OUTPUT_DIR_NAME=bart_gsc4_qa2d
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../../../src":"../../":"${PYTHONPATH}"

python run_bart_sum.py \
--data_dir=./qa2d \
--model_type=bart \
--model_name_or_path=bart-large-cnn \
--learning_rate=1e-5 \
--train_batch_size=24 \
--eval_batch_size=24 \
--output_dir=$OUTPUT_DIR \
--model_state bart_gsc/epoch=2.ckpt \
--do_train \
--do_predict \
--max_seq_length 50 \
--num_train_epochs 3 \
--early_stop_patience 1 \
--warmup_steps 600 \
--weight_decay 0.01 \
--max_grad_norm 1.0 \
--n_gpu 1
