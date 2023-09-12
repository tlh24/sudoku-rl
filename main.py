import math
import torch
import numpy as np
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
import clip_model

# convert sudoku puzzle state to a set of tokens. 
N = 9
K = 4 # how many positions to blank

sudoku = Sudoku(N, K)
sudoku.fillValues()
sudoku.printSudoku()
 
# input encoding: 
# one token for the cursor position. 
#   with it's own special label
# one token per position
# each position has:  
# a one-hot input field
# a one-hot output field
# a one-hot notes field
# three axes of position, sin/cos encoding (redundant)

cursPos = [4,4]
guess = torch.zeros(9, 9) # row, col, digit (zero = no guess)
notes = torch.zeros(9, 9, 9) # row, col, (one-hot) digit

xfrmr_width = 64
world_dim = 1 + 9*3 + 6
action_dim = 4 + 9 + 4 
	# move, digits 1-9, set/unset, note/unnote
x = torch.zeros(1 + 81, wrld_indim)

def encodePos(i, j): 
	p = torch.zeros(6)
	scl = 2 * math.pi / 9.0
	p[0] = math.sin(i*scl)
	p[1] = math.cos(i*scl)
	p[2] = math.sin(j*scl)
	p[3] = math.cos(j*scl)
	block = i // 3 + (j // 3) * 3
	p[4] = math.sin(block*scl) # slightly cheating here
	p[5] = math.cos(block*scl)
	return p

# encode the cursor token
x[0, 0] = 1
x[0, 1+9*3:] = encodePos(cursPos[0], cursPos[1])

#encode the board state
for i in range(N): 
	for j in range(N): 
		k = 1 + i*9 + j
		m = sudoku.mat[i][j] 
		if m > 0: 
			x[k, m] = 1.0
		m = guess[i][j]
		if m > 0: 
			x[k, m+9] = 1.0
		x[k, 1+9*2:1+9*3] = notes[i,j,:]
		x[k,1+9*3:] = encodePos(i, j)

plt.imshow(x.numpy())
plt.colorbar()
plt.show()


encoder = nn.Linear(wrld_indim, xfrmr_width)

# before we decode multiple actions per episode, 
# test with one action; no architectural change required. 
model = clip_model.Transformer(
	width = xfrmr_width, 
	layers = 4, 
	heads = 4, 
	attn_mask = None)

model_to_action = nn.Linear(xfrmr_width, action_dim)
model_to_reward = nn.Linear(xfrmr_width, 2) # immedate and infinite-horizon
