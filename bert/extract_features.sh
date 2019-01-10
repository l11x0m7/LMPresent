#!/bin/bash

BERT_BASE_DIR=model/uncased_L-12_H-768_A-12
python extract_features.py \
  --input_file=feature_extraction/input.txt \
  --file_type=None \
  --output_file=feature_extraction/output.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
