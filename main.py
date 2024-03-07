import math
import mmap
import random
import pickle
import torch as th
from torch import nn, optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pdb
import copy
from ctypes import * # for io
from multiprocessing import Pool
from functools import partial
import torch.multiprocessing as mp
# from lion_pytorch import Lion
from termcolor import colored

import model
from sudoku_gen import Sudoku
from plot_mmap import make_mmf, write_mmap
from constants import *
 

def actionName(act): 
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
	return sact

'''
replay_buffer is a list of lists
each sub-list = an episode. 
Eventually need to replace this with a tree for even longer planning ..
'''

class ReplayData: 
	def __init__(self, mat, cursPos, board_enc, new_board,
				  guess, notes, hotnum, hotact, reward:float, predreward:float): 
		self.mat = mat.astype(np.int32) # immutable, ref ok.
		self.cursPos = cursPos.numpy().copy().astype(np.int32) # otherwise, you store a ref.
		self.board_enc = board_enc.copy()
		self.new_board = new_board.copy()
		self.guess = guess.numpy().copy()
		self.notes = notes.numpy().copy()
		self.hotnum = hotnum.numpy().copy() # discrete: what was chosen
		self.hotact = hotact.numpy().copy()
		self.reward = reward # instant reward
		self.predreward = predreward # what we expected the lt reward to be..
	def setTotalRew(self, treward): 
		self.treward = treward
	def print(self, fd, i,j): 
		fd.write(f'[{i},{j}] cursor {self.cursPos[0]},{self.cursPos[1]}\n')
		sact = actionName(np.argmax(self.hotact))
		num = np.argmax(self.hotnum)
		fd.write(f'\t num:{num} act:{sact} rew:{self.reward}\n')

def updateNotes(cursPos, num, notes): 
	# emulate the behaviour on sudoku.com:
	# if a valid number is placed on the guess board, 
	# eliminate note possbilities accordingly
	# -- within the box 
	i,j = cursPos[0], cursPos[1]
	bi,bj = i - i%3, j - j%3
	for ii in range(3):
		for jj in range(3): 
			notes[bi+ii, bj+jj, num-1] = 0.0
	# -- within the column
	for ii in range(9):
		notes[ii, j, num-1] = 0.0
	# -- within the row
	for jj in range(9):
		notes[i, jj, num-1] = 0.0
		
def decodeAction(action): 
	pnum = th.sum(action[:10]).cpu().item()
	pact = th.sum(action[10:]).cpu().item()
	# If get badly formatted action vector, where the sum of weights are much greater/smaller than 2
	if abs(pnum + pact - 2.0) > 0.001: 
		print(colored(f'decodeAction: random choice error! {pnum},{pact}', 'cyan'))
		num = np.random.choice(10)
		act = np.random.choice(9)
	else: 
		num = np.random.choice(10, p=action[0:10].detach().cpu().numpy())
		act = np.random.choice(9, p=action[10:].detach().cpu().numpy())
	return num,act
	
def decodeActionGreedy(action): 
	num = np.argmax(action[0:10].detach().cpu().numpy())
	act = np.argmax(action[10:].detach().cpu().numpy())
	return num,act

def runAction(action, sudoku, cursPos, guess, notes): 
	'''
	Runs the action and updates the world correspondingly.
	
	Output: 
		hotnum: (tensor) One hot tensor representing the digit used, 0-indexed. 
		hotact: (tensor) One hot tensor representing the action used, 0-index
		reward: (float) Reward that results from action 
	'''
	# run the action, update the world, return the reward.
	# num, act are both 0-indexed
	num,act = decodeAction(action)
	# act = b % 4
	reward = -0.05
	if act == 0: # up
		cursPos[0] -= 1
	if act == 1: # right
		cursPos[1] += 1
	if act == 2: # down
		cursPos[0] += 1
	if act == 3: # left
		cursPos[1] -= 1
	cursPos[0] = cursPos[0] % 9 # wrap at the edges; 
	cursPos[1] = cursPos[1] % 9 # works for negative nums
	
	if act == 4: 
		clue = sudoku.mat[cursPos[0], cursPos[1]]
		curr = guess[cursPos[0], cursPos[1]]
		if sudoku.checkIfSafe(cursPos[0], cursPos[1], num) and clue == 0 and curr == 0:
			updateNotes(cursPos, num, notes)
			reward = 1 # ultimate goal is to maximize cumulative expected reward
			guess[cursPos[0], cursPos[1]] = num
		else:
			reward = -1
	if act == 5: 
		curr = guess[cursPos[0], cursPos[1]]
		if curr != 0: 
			guess[cursPos[0], cursPos[1]] = 0
		else:
			reward = -0.25
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
			reward = -0.25
	if act == 8: # do nothing. no action.
		reward = -0.075
			
	if True: 
		sact = actionName(act)
		print(f'runAction @ {cursPos[0]},{cursPos[1]}: {sact}; {num}')
	
	hotnum = th.zeros_like(action[0:10])
	hotnum[num] = 1.0
	hotact = th.zeros_like(action[10:])
	hotact[act] = 1.0
	return hotnum, hotact, reward
			
def saveReplayBuffer(replay_buffer):
	fd = open('replay_buffer.txt', 'w')
	for i,episode in enumerate(replay_buffer): 
		for j,e in enumerate(episode):
			e.print(fd, i, j)
	fd.close()
	
	fd = open('rewardlog.txt', 'w')
	for episode in replay_buffer: 
		for e in episode: 
			fd.write(f'{e.reward}\t{e.predreward}\n')
	fd.close()
	
def compressReplayBuffer(model, sudoku, replay_buffer): 
	# given input and output, infer latents to produce actions. 
	# see if this action has the same effect as the original
	# if so, replace it. 
	to_add = []
	to_remove = []
	for episode in replay_buffer: 
		board_enc = episode[0].board_enc # includes everything! cursPos etc
		new_board = episode[-1].new_board
		board_enc = board_enc.unsqueeze(0) # add a batch dim
		new_board = new_board.unsqueeze(0)
		
		latents,ap,rp = model.backLatentBoard(board_enc.cuda(), new_board.cuda())
		# check by running.
		p = episode[0]
		cursPos = th.zeros((2), dtype=th.int32)
		guess = th.zeros((9, 9))
		notes = th.ones((9, 9, 9))
		cursPos[:] = model.decodeBoardCursPos(p.board_enc) # need the init curs pos
		guess[:,:] = p.guess[:,:] 
		notes[:,:,:] = p.notes[:,:,:]
		replays = []
		sudoku.mat[:,:] = p.mat[:,:]
		for i in range(14): 
			act = ap[0, i, 10:].detach().cpu().numpy()
			if np.max(act) > 0.75: 
				board_encp = model.encodeBoard(cursPos, sudoku.mat, guess, notes)
				board_encp = th.unsqueeze(board_encp, 0)
				
				hotnum, hotact,reward = runAction(ap[0,0], sudoku, cursPos, guess, notes)
				new_boardp = model.encodeBoard(cursPos, sudoku.mat, guess, notes)
				
				board_encp = th.squeeze(board_encp) # need to store without the 
				new_boardp = th.squeeze(new_boardp) # leading batch dim
				d = ReplayData(sudoku.mat, cursPos, board_encp, new_boardp,
					guess, notes, hotnum.detach().cpu(), hotact.detach().cpu(), reward, rp[0,i,0].detach().cpu().item())
				replays.append(d)
				
		if len(replays) > 0: 
			new_board = th.squeeze(new_board)
			if th.sum(th.abs(new_board - new_boardp)) < 0.1 and len(replays) < len(episode): 
				# pdb.set_trace()
				print("!! found a replacement / simplification! !!" )
				for d in replays: 
					to_add.append(replays)
					to_remove.append(episode)
	
	for rem in to_remove: 
		try: 
			replay_buffer.remove(rem)
		except:
			print("could not remove an item from the replay buffer!")
	for add in to_add: 
		replay_buffer.append(add)

def initPuzzl(i, puzzles, sudoku, cursPos, notes): 
	'''
	Sets the notes for a particular puzzle to have a corresponding value of 0 for all non-empty squares
	and 1 for all empty squares. 
	Randomly sets a cursPos that represents an (i,j) index in a 9 by 9 square 

	notes: (tensor) Has shape (9,9,9), the last dimension represents the note digit value
	cursPos: (tensor) Has shape (2,) and is an (i,j) where i,j in [0,8]
	'''
	
	# i = np.random.randint(puzzles.shape[0])
	puzzl = puzzles[i, :, :]
	sudoku.setMat(puzzl.numpy())
	cursPos[:] = th.randint(9, (2,))
	for i in range(9):
		for j in range(9):
			if puzzl[i,j] > 0.0:
				notes[i,j,:] = 0.0 # clear all clue squares
			else: 
				notes[i,j,:] = 1.0

def fillPuzzlNotes(sudoku, notes): 
	# for each entry in the sudoku, clear corresponding notes
	for i in range(9):
		for j in range(9):
			e = sudoku.mat[i,j]
			if e > 0.0: 
				ei = int(e) - 1
				ii = i - i % 3
				jj = j - j % 3
				for k in range(9): 
					notes[i, k, ei] = 0.0
					notes[k, j, ei] = 0.0
					ik = ii + k // 3
					jk = jj + k % 3
					notes[ik,jk,ei] = 0.0


def enumerateMoves(depth, episode): 
   
	"""
	This function generates all possible sequences of moves of a certain length (depth).

	Parameters:
	depth (int): The length of the sequences of moves to generate.
	episode (list): The current sequence of moves. Initially, this should be an empty list.

	Returns:
	outlist (list): A list of all sequences of moves of length 'depth'.
	"""
    
   # Define the possible moves. In this case, there are 4 possible moves, but there are up to
	# 8 moves.
   #moves = range(8)
	moves = range(4)
	outlist = []
	if depth > 0: 
		for m in moves:
			outlist.append(episode + [m])
			outlist = outlist + enumerateMoves(depth-1, episode + [m])
	return outlist


def enumerateReplayBuffer(puzzles, model, n): 
	"""
		This function creates a replay buffer for a Sudoku game. 
		The replay buffer is a list of game states and actions, which can be used for training a model.

		Parameters:
		puzzles (tensor): List of Sudoku boards.
		model (object): The model used to encode and decode the board state.
		n (int): The number of moves to be considered.

		Returns:
		replay_buffer (list): A list of a list of ReplayData. The inner list corresponds to ReplayData
	that corresponds to each action in an episode
	"""

	all_moves = enumerateMoves(1, [])
	# If there are less than n move combos, repeat all combos list until larger
	if len(all_moves) < n: 
		rep = n // len(all_moves) + 1
		n_moves = all_moves * rep
	# If there are more than n move combos, randomly sample n combos from list
	if len(all_moves) > n: 
		n_moves = random.sample(all_moves, n)
	replay_buffer = []
	sudoku = Sudoku(9, 25)
	for i, ep in enumerate(n_moves): 
		cursPos = th.zeros((2,), dtype=th.int32)
		guess = th.zeros((9, 9))
		notes = th.ones((9, 9, 9))
		initPuzzl(i, puzzles, sudoku, cursPos, notes)
		# fillPuzzlNotes(sudoku, notes) ## NOTE FIXME (hah)
		
		# Populate a list of ReplayData that corresponds to each action in the episode
		db = []
		for act in ep: 
			num = np.random.randint(9) +1
			board_enc = model.encodeBoard(cursPos, sudoku.mat, guess, notes)
			# Store both the digit weights and the chosen action weight in an action vector 
			# Action vector: the first element is 0, the next 9 element is the weight of digits 1-9, 
			# the last 8 elements is the weight of the 8 actions
			# TODO: Change action vector to be a size 18, remove the 0 first element
			
			action = th.zeros(19)
			action[num] = 1.0 # place all weight in one digit
			action[10+act] = 1.0 # place all weight in one action
			hotnum,hotact,reward = runAction(action, sudoku, cursPos, guess, notes)
			
			# Get updated board state after executing action
			new_board = model.encodeBoard(cursPos, sudoku.mat, guess, notes)

			# # check
			# rd = model.decodeBoard(board_enc, num, act, reward)
			
			d = ReplayData(sudoku.mat, cursPos, board_enc, new_board,
						guess, notes, hotnum, hotact, reward, 0.0)
			db.append(d)
			
		replay_buffer.append(db)
		
	fid = open(f'replay_buffer_{n}.pkl', 'wb') 
	pickle.dump(replay_buffer, fid)
	fid.close()
	return replay_buffer
	
	
def runStep(sudoku, cursPos, guess, notes, reportFun): 
	# batched! 
	board_enc = th.zeros(batch_size, model.num_tokens, model.world_dim)
	for b in range(batch_size): 
		board_enc[b,:,:] = model.encodeBoard(cursPos[b,:], sudoku[b].mat, guess[b,:,:], notes[b,:,:,:])
	
	n = 16
	device = th.device(type='cuda', index=0)
	wp = th.zeros(n, batch_size, model.num_tokens, model.world_dim, device=device)
	action = th.zeros(n, batch_size, model.latent_cnt, model.action_dim, device=device)
	rp = th.zeros(n, batch_size, model.latent_cnt, 2, device=device)

	for i in range(n): 
		_, wp[i], action[i], rp[i] = model.backLatentReward(board_enc.cuda(), reportFun)
		if False:
			model.decodeBoard(board_enc[0])
			print('--- predicted new board ---')
			model.decodeBoard(wp[i,0])
			for j in range(3): 
				num,act = decodeActionGreedy(action[i,0,j,:])
				print(f'--- {actionName(act)},{num}, rew:{rp[i,0,j,0].cpu().item()} ---')
			# pdb.set_trace()
	
	rpp,indx = th.max(rp[:,:,:,0], dim=2)
	rp2,index = th.max(rpp, dim=0)
	action = action[index, range(32)]
	rp = rp[index, range(32)]
	wp = wp[index, range(32)]
	
	if True:
		model.decodeBoard(board_enc[0])
		print('--- predicted new board ---')
		model.decodeBoard(wp[0])
		for j in range(3): 
			num,act = decodeActionGreedy(action[0,j,:])
			print(f'--- {actionName(act)},{num}, rew:{rp[0,j,0].cpu().item()} ---')
		# pdb.set_trace()
	
	d_b = []
	for b in range(batch_size): 
		hotnum,hotact,reward = runAction(action[b,0,:], sudoku[b], cursPos[b], guess[b], notes[b])
		color = "black"
		if reward > 0.5: 
			color = 'red'
		print(colored(f'selected immediate reward {rp[b,0,0].detach().cpu().item()}; got {reward}', color))
		# runAction updates the cursor, notes, guess.
		new_board = model.encodeBoard(cursPos[b,:], sudoku[b].mat, guess[b,:,:], notes[b,:,:,:])
		
		d = ReplayData(sudoku[b].mat, cursPos[b,:], board_enc[b,:,:], new_board,
						guess[b], notes[b], hotnum.detach().cpu(), hotact.detach().cpu(), reward, rp[b,0,0].detach().cpu().item())
		d_b.append(d)
		
	return d_b
	
def makeBatch(replay_buffer):
	'''
	Creates a batch used for training forward prediction. Returns a batch of initial boards, 
		a batch of final boards (denoted new_boards), a batch of (possibly multiple) actions that make initial board go
		to the final board, and a batch of (possibly multiple) rewards that occur from each state, action
	
	To create an element in the batch, an episode, representing a series of actions, is randomly sampled
		and a contiguous sublist of actions is chosen from the episode. The initial board corresponds to the 
		board before the first action of the sublist and the final board corresponds to the board after
		the last action in the sublist. 

	output:
	actions_batch: (tensor) Size (batch x latent_cnt x action_dim).
		Each action from the sublist is embedded into a length action_dim, where the embed
		composes of first a one hot of the cell digit, then a one hot of the action, and then zeros
		The l-many embeddings of the sublist actions make up the first l vectors of latent_cnt, 
		then remaning are either some degenerate action or 0s 
	'''

	# Sample a random episode 
	r = np.random.randint(len(replay_buffer))
	episode = replay_buffer[r]
	
	j = np.random.randint(len(episode)) 
	# j = 0 # just use the whole episode.
	lst = episode[j:]
	
	actions_batch = th.zeros(latent_cnt, action_dim)
	rewards_batch = th.zeros(latent_cnt, reward_dim)
	for k, d in enumerate(lst):
		actions_batch[k, 0:10] = th.tensor(d.hotnum)
		actions_batch[k, 10:] = th.tensor(d.hotact)
		rewards_batch[k, 0] = th.tensor(d.reward)

	for kk in range(k+1, latent_cnt): 
		actions_batch[kk, 0] = 1.0 # zero
		actions_batch[kk, -1] = 1.0 # noop 
	# First element of reward vector is the actual reward (floating num), second element is the cumulative reward
	rewards_batch[:,1] = th.cumsum(rewards_batch[:,0], dim=0)
	d = lst[0] # starting at index j, see above.
	board_batch = th.tensor(d.board_enc)
	d = lst[-1]
	new_board_batch = th.tensor(d.new_board)

	return board_batch, new_board_batch, actions_batch, rewards_batch
	
if __name__ == '__main__':
	fd = open('rewardlog.txt', 'w') # truncate the file
	fd.write(f'{0}\t{0}\n')
	fd.close()
	
	# th.cuda.memory._record_memory_history()
	
	sudoku = [Sudoku(9, 25) for _ in range(batch_size)]

	model = model.Racoonizer(
		xfrmr_dim = xfrmr_dim, 
		world_dim = world_dim,
		latent_cnt = latent_cnt, 
		latent_dim = latent_dim,
		action_dim = action_dim, 
		reward_dim = reward_dim).cuda()
	
	mp.set_start_method('spawn')
	puzzles = th.load('puzzles_500000.pt')
	n = 44000 # no notes.
	# Reads saved replay buffer file if exists, else generates a replay buffer
	try: 
		fname = f'replay_buffer_{n}.pkl'
		fid = open(fname, 'rb') 
		print(f'loading {fname}')
		replay_buffer = pickle.load(fid)
		fid.close()
	except: 
		replay_buffer = enumerateReplayBuffer(puzzles, model, n)
	# saveReplayBuffer(replay_buffer)
	
	model.printParamCount()
	try: 
		model.load_checkpoint()
	except: 
		print("could not load the model parameters.")
	
	fd_board = make_mmf("board.mmap", [batch_size, 82, world_dim])
	fd_new_board = make_mmf("new_board.mmap", [batch_size, 82, world_dim])
	fd_worldp = make_mmf("worldp.mmap", [batch_size, 82, world_dim])
	fd_action = make_mmf("action.mmap", [batch_size, latent_cnt, action_dim])
	fd_actionp = make_mmf("actionp.mmap", [batch_size, latent_cnt, action_dim])
	fd_reward = make_mmf("reward.mmap", [batch_size, latent_cnt, reward_dim])
	fd_rewardp = make_mmf("rewardp.mmap", [batch_size, latent_cnt, reward_dim])
	fd_latent = make_mmf("latent.mmap", [batch_size, latent_cnt, action_dim+latent_dim])
	fd_attention = make_mmf("attention.mmap", [2, token_cnt, token_cnt, n_heads])
	fd_wqkv = make_mmf("wqkv.mmap", [n_heads*2,2*xfrmr_dim,xfrmr_dim])

	fd_losslog = open('losslog.txt', 'w')
	uu = 0
	
	cursPos = th.zeros((batch_size, 2), dtype=th.int32)
	guess = th.zeros((batch_size, 9, 9))
	notes = th.ones((batch_size, 9, 9, 9))
	episodes = [[] for _ in range(batch_size)]
	
	optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay = 1e-2)
	# optimizer = Lion(model.parameters(), lr=7e-5, weight_decay = 7e-3)
	# optimizer = optim.Adagrad(model.parameters(), lr=3e-4, weight_decay = 0.0)
	# optimizer = optim.Rprop(model.parameters(), lr=3e-3)
		
	epoch = 700 # was 700
	priosiz = epoch * batch_size
	device = th.device(type='cuda', index=0)
	prio_board = th.zeros(priosiz, 82, world_dim, device=device)
	prio_new_board = th.zeros(priosiz, 82, world_dim, device=device)
	prio_actions = th.zeros(priosiz, latent_cnt, action_dim, device=device)
	prio_rewards = th.zeros(priosiz, latent_cnt, reward_dim, device=device)
	prio_rewardp = th.zeros(priosiz, latent_cnt, reward_dim, device=device)
	prio_loss = th.zeros(priosiz)
	prio_valid = th.zeros(priosiz)
		
	for p in range(2000): # 100 is enough dec20
# 		for b in range(batch_size): 
# 			i = np.random.randint(puzzles.shape[0])
# 			initPuzzl(i, puzzles, sudoku[b], cursPos[b], notes[b])
# 		for u in range(50): 
# 			db = runStep(sudoku, cursPos, guess, notes, reportFun) 
# 			for b in range(batch_size):
# 				d = db[b]
# 				episodes[b].append(d)
# 				
# 				if d.reward < -0.5 or len(episodes[b]) > 13:
# 					replay_buffer.append(episodes[b])
# 					episodes[b] = []
# 					i = np.random.randint(puzzles.shape[0])
# 					initPuzzl(i, puzzles, sudoku[b], cursPos[b], notes[b])

		# saveReplayBuffer(replay_buffer)
		
		# if p % 10 == 0: 
		# 	oldlen = len(replay_buffer)
		# 	compressReplayBuffer(model, sudoku[0], replay_buffer)
		# 	newlen = len(replay_buffer)
		# 	print(f'replay buffer old {oldlen} to {newlen}')

		# TODO: 
		# -- Start some games from the middle
		# -- Select actions for **more than just reward** 
		#    maybe predictability, estimated information gain, 
		#    some other internal metrics?  
		#    some sort of internally generated progress, 
		#    inclusive of information gain?  
		# -- prune rollouts by total reward: ignore actions that just cost time.
		#    need to avoid degeneracy: sampling the same option over and over
		#    internal novelty reward? 
		# -- ignore equivalences in rollouts: 
		#    model should predict simpler actions!!
		# -- add in option to sample multipe actions?? if they are predictable?
		# -- continue to check the board predictions etc.  
		# -- verify that it's actually converging (half check)
		# -- can memorize the training dataset (mostly check)
		# -- run it on the GPU (check)
		# -- select longer runs for prediction-training
		# -- prune away useless rollouts?
		
		for u in range(epoch//2): 
			
			board = th.zeros(batch_size, 82, world_dim)
			new_board = th.zeros(batch_size, 82, world_dim)
			actions = th.zeros(batch_size, latent_cnt, action_dim)
			rewards = th.zeros(batch_size, latent_cnt, reward_dim)
			
			for b in range(batch_size): 
				board[b,:,:], new_board[b,:,:], actions[b,:,:], rewards[b,:] = makeBatch(replay_buffer)
				
			board = board.cuda()
			new_board = new_board.cuda()
			actions = actions.cuda()
			rewards = rewards.cuda()
			# reportFun will write the rest. 
			
			# latents = model.backLatent(board, new_board, actions, reportFun)
			# pdb.set_trace()
			latents = actions[:,:,:] # "cheat" to generate structure.
			
			model.zero_grad()
			wp, ap, rp, a1, a2, w1, w2 = model.forward(board, latents, uu)
			
			# keep the batch dim
			loss = th.sum((new_board - wp)**2, (1,2))*0.15 + \
						th.sum((actions - ap)**2, (1,2)) + \
						(rewards[:,0,0] - rp[:,0,0])**2 * 4.0
			lossall = th.sum(loss)
			lossall.backward()
			# th.nn.utils.clip_grad_norm_(model.parameters(), 1.0) -- doesn't work!?
			optimizer.step() 
			
			lossall.detach()
			wqkv1 = th.sum(model.xfrmr.layer1.wqkv.weight).cpu().detach().item()
			wqkv2 = th.sum(model.xfrmr.layer2.wqkv.weight).cpu().detach().item()
			print(lossall.cpu().item(), wqkv1, wqkv2)
			fd_losslog.write(f'{uu}\t{lossall.cpu().item()}\n')
			fd_losslog.flush()
			uu = uu + 1

			if uu % 24 == 0: 
				write_mmap(fd_board, board.cpu())
				write_mmap(fd_new_board, new_board.cpu())
				write_mmap(fd_action, actions.cpu())
				write_mmap(fd_reward, rewards.cpu())
				
				write_mmap(fd_worldp, wp.cpu().detach())
				write_mmap(fd_actionp, ap.cpu().detach())
				write_mmap(fd_rewardp, rp.cpu().detach())
				lslow = model.latent_slow.unsqueeze(0).expand(batch_size, -1, -1)
				write_mmap(fd_latent, th.cat((latents, lslow), 2).cpu().detach())
				write_mmap(fd_attention, th.stack((a1, a2), 0))
				write_mmap(fd_wqkv, th.stack((w1, w2), 0))
				
				i = np.random.randint(batch_size)
				r = rewards[i,0,0].cpu().item()
				# num,act = decodeAction(actions[0,0,:].cpu())
				# rd = model.decodeBoard(board[0,:,:].cpu().numpy(), num,act,r) # positive control!
				rd = rp[i,0,0].cpu().item()
				
				fd = open('rewardlog.txt', 'a')
				fd.write(f'{r}\t{rd}\n')
				fd.close()
			
			# prioritized replay buffer action!
			indx = (epoch//2 + u) * batch_size 
				# add to the end, replace accurate examples
			indy = indx + batch_size
			prio_board[indx:indy, :, :] = board
			prio_new_board[indx:indy, :, :] = new_board
			prio_actions[indx:indy, :, :] = actions
			prio_rewards[indx:indy, :, :] = rewards
			prio_rewardp[indx:indy, :, :] = rp
			prio_loss[indx:indy] = loss.detach()
			prio_valid[indx:indy] = 1.0
			
		# run over the whole prioritized repaly buffer, update the loss. 
		# allow for warm-up to find the challenging examples. 
		qrange = 1 + uu // 3500
		qrange = min(qrange, 3)
		print(f'q range {qrange}')
		for q in range(qrange): 
			srt = th.argsort(prio_loss*prio_valid, descending=True) # stable=True
			prio_board = prio_board[srt, :, :]
			prio_new_board = prio_new_board[srt, :, :]
			prio_actions = prio_actions[srt, :, :]
			prio_rewards = prio_rewards[srt, :, :]
			prio_rewardp = prio_rewardp[srt, :, :]
			prio_loss = prio_loss[srt]
			prio_valid = prio_valid[srt]
			
			x1 = prio_loss.cpu()
			x2 = prio_rewards[:,0,0].cpu().numpy()
			x3 = prio_rewardp[:,0,0].cpu().detach().numpy()
			y = np.stack((x1,x2,x3), axis=0)
			np.save('prio.npy', y)
		
			for u in range(math.floor(epoch * 0.75)): 
				indx = u * batch_size
				indy = indx + batch_size
				board = prio_board[indx:indy, :, :]
				new_board = prio_new_board[indx:indy, :, :]
				actions = prio_actions[indx:indy, :, :]
				rewards = prio_rewards[indx:indy, :, :]
				
				latents = actions[:,:,:] # "cheat" to generate structure.
				
				model.zero_grad()
				wp, ap, rp, a1,a2,w1,w2 = model.forward(board, latents, uu)
				
				# keep the batch dim
				loss = th.sum((new_board - wp)**2, (1,2))*0.15 + \
							th.sum((actions - ap)**2, (1,2)) + \
							(rewards[:,0,0] - rp[:,0,0])**2
				lossall = th.sum(loss)
				lossall.backward()
				#th.nn.utils.clip_grad_norm_(model.parameters(), 0.025)
				optimizer.step() 
				
				prio_loss[indx:indy] = loss.detach() # update loss. 
				
				if u == 0: 
					write_mmap(fd_board, board.cpu())
					write_mmap(fd_new_board, new_board.cpu())
					write_mmap(fd_action, actions.cpu())
					write_mmap(fd_reward, rewards.cpu())
					
					write_mmap(fd_worldp, wp.cpu().detach())
					write_mmap(fd_actionp, ap.cpu().detach())
					write_mmap(fd_rewardp, rp.cpu().detach())
					lslow = model.latent_slow.unsqueeze(0).expand(batch_size, -1, -1)
					write_mmap(fd_latent, th.cat((latents, lslow), 2).cpu().detach())
					write_mmap(fd_attention, th.stack((a1, a2), 0))
					write_mmap(fd_wqkv, th.stack((w1, w2), 0))
					
					r = rewards[0,0,0].cpu().item()
					# num,act = decodeAction(actions[0,0,:].cpu())
					# rd = model.decodeBoard(board[0,:,:].cpu().numpy(), num,act,r) # positive control!
					rd = rp[0,0,0].cpu().item()
					
					fd = open('rewardlog.txt', 'a')
					fd.write(f'{r}\t{rd}\n')
					fd.close()
		
				
		# model.save_checkpoint()

	fd_losslog.close()
	pdb.set_trace()
	# th.cuda.memory._dump_snapshot("my_snapshot.pickle")
