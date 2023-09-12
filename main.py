import math
import torch
import numpy as np
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
import model
 
# input encoding: 
# one token for the cursor position. 
#   with it's own special label
# one token per position
# each position has:  
# a one-hot input field
# a one-hot output field
# a one-hot notes field
# three axes of position, sin/cos encoding (redundant)

# cursPos = [4,4]
# guess = torch.zeros(9, 9) # row, col, digit (zero = no guess)
# notes = torch.zeros(9, 9, 9) # row, col, (one-hot) digit

sudoku = Sudoku(9, 25)

xfrmr_width = 64
world_dim = 1 + 9*3 + 6 # must be even!
action_dim = 10 + 8 # digits 0-9 (0=nothing); move, set/unset, note/unnote
reward_dim = 2 # immediate and infinte-horizon
latent_cnt = 96 - 81 - 1


# before we decode multiple actions per episode, 
# test with one action; no architectural change required. 
model = model.Racoonizer(
	xfrmr_width = xfrmr_width, 
	world_dim = world_dim,
	latent_cnt = latent_cnt, 
	action_dim = action_dim, 
	reward_dim = reward_dim)

# board_enc = model.encodeBoard(cursPos, sudoku.mat, guess, notes)

# for i in range(5): 
# 	latents = torch.randn(latent_cnt, world_dim // 2) * 0.25
# 	act, rew = model.forward(board_enc, latents)
# 	print(act, act.sum(), rew)


def runAction(action, sudoku, cursPos, guess, notes): 
	# run the action, update the world, return the reward.
	num = np.random.choice(10, p=action[0:10].detach().numpy())
	act = np.random.choice(8, p=action[10:].detach().numpy())
	reward = 0
	if act == 0: # up
		cursPos[0] -= 1
	if act == 1: # right
		cursPos[1] += 1
	if act == 2: # down
		cursPos[0] += 1
	if act == 3: # left
		cursPos[1] -= 1
	cursPos[0] = cursPos[0] % 9
	cursPos[1] = cursPos[1] % 9
	
	if act == 4: 
		curr = guess[cursPos[0], cursPos[1]]
		if sudoku.checkIfSafe(cursPos[0], cursPos[1], num) and curr == 0: 
			reward = 1 # ultimate goal is to maximize cumulative expected reward
			guess[cursPos[0], cursPos[1]] = num
		else:
			reward = -1
	if act == 5: 
		guess[cursPos[0], cursPos[1]] = 0
	# no reward for notes -- this has to be algorithmic
	if act == 6: 
		if notes[cursPos[0], cursPos[1], num] == 0:
			notes[cursPos[0], cursPos[1], num] = 1.0
		else: 
			reward = -1 # penalize redundant actions
	if act == 7: 
		if notes[cursPos[0], cursPos[1], num] > 0:
			notes[cursPos[0], cursPos[1], num] = 0.0
		else:
			reward = -1
			
	if True: 
		sact = '-'
		if act == 0: 
			sact = 'up'
		if act == 1: 
			sact = 'right'
		if act == 2:
			sact = 'down'
		if act == 3: 
			sact = 'left'
		if act == 4: 
			sact = 'set guess'
		if act == 5:
			sact = 'unset guess'
		if act == 6:
			sact = 'set note'
		if act == 7:
			sact = 'unset note'
		print(f'runAction @ {cursPos[0]},{cursPos[1]}: {sact}; {num}')
	
	return num, act, reward
	
def runStep(sudoku, cursPos, guess, notes, previous): 
	board_enc = model.encodeBoard(cursPos, sudoku.mat, guess, notes)
	if False: 
		plt.imshow(board_enc.numpy())
		plt.colorbar()
		plt.show()

	
	latents = torch.randn(latent_cnt, world_dim // 2) * 0.25
	# need to do a little optimization here.. ? 
	action, pred_rew = model.forward(board_enc, latents)
	
	num,act,reward = runAction(action, sudoku, cursPos, guess, notes)
	
	d = ReplayData(sudoku.mat, cursPos, board_enc, 
					guess, notes, num, act, reward, previous)
	return d


class ReplayData: 
	def __init__(self, mat, curs_pos, board_enc, 
				  guess, notes, num, act, reward, previous): 
		self.mat = mat
		self.curs_pos = curs_pos
		self.board_enc = board_enc
		self.guess = guess
		self.num = num # discrete: what was chosen
		self.act = act
		self.reward = reward
		self.previous = previous # what was the previous action? 
			# needed for updating the cumulative reward. 

replay_buffer = []
puzzles = torch.load('puzzles_100000.pt')

for u in range(10):
	i = np.random.randint(puzzles.shape[0])
	puzzl = puzzles[i, :, :]
	sudoku.setMat(puzzl.numpy())
	
	cursPos = [4,4]
	guess = torch.zeros(9, 9) # row, col, digit (zero = no guess)
	notes = torch.zeros(9, 9, 9) # row, col, (one-hot) digit
	previous = -1
	
	d = runStep(sudoku, cursPos, guess, notes, previous)
	
	previous = len(replay_buffer)
	replay_buffer.append(d)
	
	v = 0
	while d.reward >= 0 and v < 10:
		d = runStep(sudoku, cursPos, guess, notes, previous)
		previous = len(replay_buffer)
		replay_buffer.append(d)
		v += 1
	


