import math
import torch
import numpy as np
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
import model
from multiprocessing import Pool
from itertools import product

# the naive sudoku generator is slow. 
# parallelize and save the results for fast loading later.

N = 500000

x = torch.zeros(N, 9, 9)

def makePuzzle(j): # argument is ignored.
	k = np.random.randint(20) + 25 # how many positions to blank.
	sudoku = Sudoku(9, k)
	sudoku.fillValues()
	return torch.tensor(sudoku.mat)


pool = Pool() #defaults to number of available CPU's
chunksize = 1 # some puzzles require a lot of backtracking to fill, so keep small
for ind, res in enumerate(pool.imap_unordered(makePuzzle, range(N), chunksize)):
	x[ind, :, :] = res


torch.save(x, f'puzzles_{N}.pt')
