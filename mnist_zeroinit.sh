#!/bin/bash

train_size=1000

python mnist_zeroinit.py --train-size $train_size -a -z -c 0 &
python mnist_zeroinit.py --train-size $train_size -a -z -c 0 &
python mnist_zeroinit.py --train-size $train_size -a -z -c 1 &
python mnist_zeroinit.py --train-size $train_size -a -z -c 1 &

wait

python mnist_zeroinit.py --train-size $train_size -a -c 0 &
python mnist_zeroinit.py --train-size $train_size -a -c 0 &
python mnist_zeroinit.py --train-size $train_size -a -c 1 &
python mnist_zeroinit.py --train-size $train_size -a -c 1 &
