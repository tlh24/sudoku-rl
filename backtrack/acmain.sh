#!/usr/bin/bash
for i in {1..20}
do
./backup.sh
parallel -j4 -u --link 'python acmain.py -b 64 --puzz 1 -r 2 -i {1} --cuda {2}' ::: 3 2 1 0 ::: 0 0 1 1 # ashtray
# parallel -j8 -u --link 'python acmain.py -b 64 --puzz 1 -r 2 -i {1} --cuda {2}' ::: 0 1 2 3 4 5 6 7 ::: 3 2 1 0 3 2 1 0 # 4-gpu server
python acmain.py -r 2
done
