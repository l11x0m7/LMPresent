#!/bin/bash

python2 train.py --dataset race --desc race --submit --data_dir ../data/RACE/RACE

python2 test.py --dataset race --desc race --data_dir ../data/RACE/RACE
