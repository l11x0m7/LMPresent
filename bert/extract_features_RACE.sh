#!/bin/bash

BERT_BASE_DIR=model/uncased_L-12_H-768_A-12

# train data
python extract_features.py \
  --input_file=../data/RACE/RACE/train/high \
  --file_type=RACE \
  --output_file=../data/RACE/features_train_high.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=512 \
  --max_seq_length_for_opts=64 \
  --batch_size=8


# dev data
python extract_features.py \
  --input_file=../data/RACE/RACE/dev/high \
  --file_type=RACE \
  --output_file=../data/RACE/features_dev_high.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=512 \
  --max_seq_length_for_opts=64 \
  --batch_size=8


# test data
python extract_features.py \
  --input_file=../data/RACE/RACE/test/high \
  --file_type=RACE \
  --output_file=../data/RACE/features_test_high.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=512 \
  --max_seq_length_for_opts=64 \
  --batch_size=8



# train data
python extract_features.py \
  --input_file=../data/RACE/RACE/train/middle \
  --file_type=RACE \
  --output_file=../data/RACE/features_train_middle.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=512 \
  --max_seq_length_for_opts=64 \
  --batch_size=8


# dev data
python extract_features.py \
  --input_file=../data/RACE/RACE/dev/middle \
  --file_type=RACE \
  --output_file=../data/RACE/features_dev_middle.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=512 \
  --max_seq_length_for_opts=64 \
  --batch_size=8


# test data
python extract_features.py \
  --input_file=../data/RACE/RACE/test/middle \
  --file_type=RACE \
  --output_file=../data/RACE/features_test_middle.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=512 \
  --max_seq_length_for_opts=64 \
  --batch_size=8