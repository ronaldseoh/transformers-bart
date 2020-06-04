# Instructions to train the BART neural conversion model

1. Add transformers library to PYTHONPATH

2. 
```
summarization/bart/run_bart_sum.py \
--data_dir=.data/google_sentence_compression \
--model_type=bart \
--model_name_or_path=bart-large-cnn \
--learning_rate=1e-5 \
--train_batch_size=24 \
--eval_batch_size=24 \
--output_dir=bart_gsc \
--do_train \
--do_predict \
--max_seq_length 50 \
--num_train_epochs 3 \
--early_stop_patience 1 \
--warmup_steps 1125 \
--weight_decay 0.01 \
--max_grad_norm 1.0 \
--n_gpu 1
```

```
summarization/bart/run_bart_sum.py \
--data_dir=.data/qa_to_hypothesis \
--model_type=bart \
--model_name_or_path=bart-large-cnn \
--learning_rate=1e-5 \
--train_batch_size=24 \
--eval_batch_size=24 \
--output_dir=bart_gsc4_qa2d\
--model_state bart_gsc/epoch=1.ckpt \
--do_train \
--do_predict \
--max_seq_length 50 \
--num_train_epochs 3 \
--early_stop_patience 1 \
--warmup_steps 600 \
--weight_decay 0.01 \
--max_grad_norm 1.0 \
--n_gpu 1
```
