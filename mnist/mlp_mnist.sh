#!/usr/bin/bash
python mlp_mnist.py --hidden 128 --reset
python mlp_mnist.py --hidden 256
python mlp_mnist.py --hidden 512
python mlp_mnist.py --hidden 1024
python mlp_mnist.py --hidden 2048
python mlp_mnist.py --hidden 3072
python mlp_mnist.py --hidden 4096
python mlp_mnist.py --hidden 6144
python mlp_mnist.py --hidden 8192

