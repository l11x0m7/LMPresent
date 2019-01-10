#!/bin/bash

export BERT_BASE_DIR=model/uncased_L-12_H-768_A-12
export RACE_DIR=../data/RACE/RACE

python run_classifier_RACE.py \
  --task_name=RACE2 \
  --do_train=true \
  --do_eval=true \
  --data_dir=$RACE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/race_output/
