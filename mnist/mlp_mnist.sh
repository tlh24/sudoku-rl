#!/usr/bin/bash
python mlp_mnist.py --hidden 512
python mlp_mnist.py --hidden 1024
python mlp_mnist.py --hidden 2048
python mlp_mnist.py --hidden 4096
python mlp_mnist.py --hidden 8192

