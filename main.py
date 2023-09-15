import math
import mmap
import torch as th
from torch import nn, optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pdb
from ctypes import * # for io

import model
from sudoku_gen import Sudoku
from plot_mmap import make_mmf, write_mmap
from constants import *
 
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
# guess = th.zeros(9, 9) # row, col, digit (zero = no guess)
# notes = th.zeros(9, 9, 9) # row, col, (one-hot) digit

sudoku = Sudoku(9, 25)

# before we decode multiple actions per episode, 
# test with one action; no architectural change required. 
model = model.Racoonizer(
	xfrmr_width = xfrmr_width, 
	world_dim = world_dim,
	latent_cnt = latent_cnt, 
	action_dim = action_dim, 
	reward_dim = reward_dim)

# board_enc = model.encodeBoard(cursPos, sudoku.mat, guess, notes)

class ReplayData: 
	def __init__(self, mat, curs_pos, board_enc, new_board,
				  guess, notes, hotnum, hotact, reward): 
		self.mat = mat
		self.curs_pos = curs_pos
		self.board_enc = board_enc
		self.new_board = new_board
		self.guess = guess
		self.hotnum = hotnum # discrete: what was chosen
		self.hotact = hotact
		self.reward = reward # instant reward
		self.previous = -1
	def setPrev(self, previous): 
		self.previous = previous
	def setTotalRew(self, treward): 
		self.treward = treward

def updateNotes(cursPos, num, notes): 
	# emulate the behaviour on sudoku.com:
	# if a valid number is placed on the guess board, 
	# eliminate note possbilities accordingly
	# -- within the box 
	i,j = cursPos[0],cursPos[1]
	bi,bj = i - i%3, j - j%3
	for ii in range(3):
		for jj in range(3): 
			notes[bi+ii][bj+jj][num-1] = 0.0
	# -- within the column
	for ii in range(9):
		notes[ii][j][num-1] = 0.0
	# -- within the row
	for jj in range(9):
		notes[i][jj][num-1] = 0.0

def runAction(action, sudoku, cursPos, guess, notes): 
	# run the action, update the world, return the reward.
	i = 0
	num = np.random.choice(10, p=action[i,0:10].detach().numpy())
	act = np.random.choice(9, p=action[i,10:].detach().numpy())
	reward = 0
	if act == 0: # up
		cursPos[0] -= 1
		reward = -0.05
	if act == 1: # right
		cursPos[1] += 1
		reward = -0.05
	if act == 2: # down
		cursPos[0] += 1
		reward = -0.05
	if act == 3: # left
		cursPos[1] -= 1
		reward = -0.05
	cursPos[0] = cursPos[0] % 9
	cursPos[1] = cursPos[1] % 9
	
	if act == 4: 
		curr = guess[cursPos[0], cursPos[1]]
		if sudoku.checkIfSafe(cursPos[0], cursPos[1], num) and curr == 0:
			updateNotes(cursPos, num, notes)
			reward = 1 # ultimate goal is to maximize cumulative expected reward
			guess[cursPos[0], cursPos[1]] = num
		else:
			reward = -1
	if act == 5: 
		guess[cursPos[0], cursPos[1]] = 0
	# no reward/cost for notes -- this has to be algorithmic/inferred
	if act == 6: 
		if notes[cursPos[0], cursPos[1], num-1] == 0:
			notes[cursPos[0], cursPos[1], num-1] = 1.0
		else: 
			reward = -0.25 # penalize redundant actions
	if act == 7: 
		if notes[cursPos[0], cursPos[1], num-1] > 0:
			notes[cursPos[0], cursPos[1], num-1] = 0.0
		else:
			reward = -1
	if act == 8: # do nothing. no action.
		reward = -0.04
			
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
		if act == 8: 
			sact = 'nop'
		print(f'runAction @ {cursPos[0]},{cursPos[1]}: {sact}; {num}')
	
	hotnum = th.zeros_like(action[i,0:10])
	hotnum[num] = 1.0
	hotact = th.zeros_like(action[i,10:])
	hotact[act] = 1.0
	return hotnum, hotact, reward
	
def runStep(sudoku, cursPos, guess, notes): 
	board_enc = model.encodeBoard(cursPos, sudoku.mat, guess, notes)
	board_enc = th.unsqueeze(board_enc, 0)

	selec = []
	for i in range(6):
		latents = th.randn(1, latent_cnt, world_dim // 2)
		_, action, pred_rew = model.forward(board_enc, latents)
		selec.append(( pred_rew[0,0,1], action.detach(), latents.detach() ))
	selec = sorted(selec, key=lambda s: -1*s[0])
	ltrew,action,latents = selec[0]
	print(f'selected long term reward {ltrew}')
	action = th.squeeze(action)
	# discard latents -- re-estimate later (depends on model)
	
	hotnum,hotact,reward = runAction(action, sudoku, cursPos, guess, notes)
	# runAction updates the cursor, notes, guess.
	new_board = model.encodeBoard(cursPos, sudoku.mat, guess, notes)
	
	d = ReplayData(sudoku.mat, cursPos, board_enc, new_board,
					guess, notes, hotnum, hotact, reward)
	return d


replay_buffer = [] # this needs to be a tree. 
puzzles = th.load('puzzles_100000.pt')

def updateTotalRew(): 
	# for each element in the replay buffer, 
	# update the total reward with the maximum reward  
	# for any path eminating from a given node.
	for e in replay_buffer: 
		e.setTotalRew(0.0)
	def update(e):
		p = e.previous
		r = 0.0
		if p >= 0:
			r = e.reward + update(replay_buffer[p])
		else:
			r = e.reward
		return r
	for e in replay_buffer: 
		r = update(e)
		if r > e.treward:
			e.setTotalRew(r)
			
fd_board = make_mmf("board.mmap", [batch_size, 82, world_dim])
fd_new_board = make_mmf("new_board.mmap", [batch_size, 82, world_dim])
fd_worldp = make_mmf("worldp.mmap", [batch_size, 82, world_dim])
fd_action = make_mmf("action.mmap", [batch_size, latent_cnt, action_dim])
fd_actionp = make_mmf("actionp.mmap", [batch_size, latent_cnt, action_dim])
fd_reward = make_mmf("reward.mmap", [batch_size, latent_cnt, reward_dim])
fd_rewardp = make_mmf("rewardp.mmap", [batch_size, latent_cnt, reward_dim])

		
for p in range(100): 
	for u in range(100):
		i = np.random.randint(puzzles.shape[0])
		puzzl = puzzles[i, :, :]
		sudoku.setMat(puzzl.numpy())
		
		cursPos = [np.random.randint(9),np.random.randint(9)]
		guess = th.zeros(9, 9) # row, col, digit (zero = no guess)
		notes = th.ones(9, 9, 9) # row, col, (one-hot) digit
		for i in range(9):
			for j in range(9):
				if puzzl[i,j] > 0.0:
					notes[i,j,:] = 0.0 # clear all clue squares
		
		d = runStep(sudoku, cursPos, guess, notes)
		replay_buffer.append(d)
		
		v = 0
		while d.reward >= 0 and v < 14:
			dp = runStep(sudoku, cursPos, guess, notes)
			dp.setPrev(len(replay_buffer)-1)
			replay_buffer.append(dp)
			d = dp
			v += 1

	updateTotalRew()
		
	# TODO: need to start some games from the middle .. 
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	# TODO: 
	# -- prune rollouts by total reward: ignore actions that just cost time.
	#    need to avoid degeneracy: sampling the same option over and over
	# -- add in option to sample multipe actions?? if they are predictable?
	# -- continue to check the board predictions etc.  
	# -- verify that it's actually converging 
	# -- can memorize the training dataset
	# -- run it on the GPU
	# -- select longer runs for prediction-training
	# -- prune away useless rollouts?
	for u in range(100): 
		board = th.zeros(batch_size, 82, world_dim)
		new_board = th.zeros(batch_size, 82, world_dim)
		actions = th.zeros(batch_size, latent_cnt, action_dim)
		rewards = th.zeros(batch_size, latent_cnt, reward_dim)
		latents = th.zeros(batch_size, latent_cnt, world_dim//2)
		
		for b in range(batch_size): 
			i = np.random.randint(len(replay_buffer))
			d = replay_buffer[i]
			board[b,:,:] = d.board_enc
			new_board[b,:,:] = d.new_board
			# NOTE: need to sample a variable number of output actions
			actions[b, 0, 0:10] = d.hotnum
			actions[b, 0, 10:] = d.hotact
			rewards[b, 0,0] = d.reward
			rewards[b, 0,1] = d.treward
			

		latents = model.backLatent(board, new_board, actions, rewards)
		
		wp, ap, rp = model.forward(board, latents)
		loss = th.sum((new_board - wp)**2)*0.05 + \
					th.sum((actions - ap)**2) + \
					th.sum((rewards - rp)**2) 
		loss.backward()
		#th.nn.utils.clip_grad_norm_(model.parameters(), 0.025)
		optimizer.step() 
		
		loss.detach()
		print(loss.item())

		if u % 10 == 9: 
			write_mmap(fd_board, board)
			write_mmap(fd_new_board, new_board)
			write_mmap(fd_worldp, wp.detach())
			write_mmap(fd_action, actions)
			write_mmap(fd_actionp, ap.detach())
			write_mmap(fd_reward, rewards)
			write_mmap(fd_rewardp, rp.detach())
