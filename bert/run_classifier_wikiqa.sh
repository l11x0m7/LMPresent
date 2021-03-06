#!/bin/bash

export BERT_BASE_DIR=~/models/uncased_L-12_H-768_A-12
export WIKI_DIR=~/data/WikiQACorpus

python3 run_classifier_wikiqa.py \
  --task_name=wikiqa \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$WIKI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/wikiqa_output/
