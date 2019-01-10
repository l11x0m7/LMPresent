SQUAD_DIR=../data/squad
output_dir=/tmp/squad_base2


python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json $output_dir/predictions.json --na-prob-file $output_dir/null_odds.json
