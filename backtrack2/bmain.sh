#!/usr/bin/bash
# python bmain.py -i 0 --cuda 0 &
# python bmain.py -i 1 --cuda 0 &
# python bmain.py -i 2 --cuda 1 &
# python bmain.py -i 3 --cuda 1 & wait
parallel -j4 -u --link 'python bmain.py -i {1} --cuda {2}' ::: 3 2 1 0 ::: 0 0 1 1
