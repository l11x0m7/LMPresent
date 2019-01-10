SQUAD_DIR=../data/squad
output_dir=/tmp/squad_base


python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $output_dir/predictions.json
