import math
import torch
import numpy as np
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product

# the naive sudoku generator is slow. 
# parallelize and save the results for fast loading later.

N = 500000
S = 4 # normally 9

x = torch.zeros(N, S, S)

def makePuzzle(j): # argument is ignored.
	if S == 9: 
		k = np.random.randint(20) + 25 # how many positions to blank.
	if S == 4: 
		k = np.random.randint(5) + 6
	sudoku = Sudoku(S, k)
	sudoku.fillValues()
	return torch.tensor(sudoku.mat)


pool = Pool() #defaults to number of available CPU's
chunksize = 1 # some puzzles require a lot of backtracking to fill, so keep small
for ind, res in enumerate(pool.imap_unordered(makePuzzle, range(N), chunksize)):
	x[ind, :, :] = res


torch.save(x, f'puzzles_{S}_{N}.pt')
