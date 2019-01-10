# finetune-transformer-lm on RACE

Code and model for the paper "Improving Language Understanding by Generative Pre-Training"

Currently I rerun this code on RACE dataset and the training code is shown as follows:

`python train.py --dataset race --desc race --submit --data_dir [path to RACE dataset]`

the test code is shown below:

`python2 test.py --dataset race --desc race --data_dir [path to RACE dataset]`

