#!/usr/bin/bash
for i in {1..20}
do
./backup.sh
parallel -j4 -u --link 'python acmain.py -b 64 --puzz 3 -r 2 -i {1} --cuda {2}' ::: 3 2 1 0 ::: 3 2 1 0
python acmain.py -r 2
done
