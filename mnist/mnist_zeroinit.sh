#!/usr/bin/bash

ts=100

python mnist_zeroinit.py --train-size $ts --adagrad -c 0 --hidden 4 &
python mnist_zeroinit.py --train-size $ts --adagrad -c 1 -z --hidden 4 &
wait
python mnist_zeroinit.py --train-size $ts --adagrad -c 0 --hidden 4 &
python mnist_zeroinit.py --train-size $ts --adagrad -c 1 -z --hidden 4 &
wait

# python mnist_zeroinit.py --train-size $train_size -a -c 0 &
# python mnist_zeroinit.py --train-size $train_size -a -c 0 &
# python mnist_zeroinit.py --train-size $train_size -a -c 1 &
# python mnist_zeroinit.py --train-size $train_size -a -c 1 &
# 
# wait
