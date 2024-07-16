import math
import random
import argparse
import time
import os
import glob # for file filtering
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pdb
from termcolor import colored
import pickle
import matplotlib.pyplot as plt
import sparse_encoding
from gracoonizer import Gracoonizer
from sudoku_gen import Sudoku
from plot_mmap import make_mmf, write_mmap
from netdenoise import NetDenoise
from test_gtrans import getTestDataLoaders, SimpleMLP
from constants import *
from type_file import Action, Axes, getActionName
from l1attn_sparse_cuda import expandCoo
import psgd 
	# https://sites.google.com/site/lixilinx/home/psgd
	# https://github.com/lixilinx/psgd_torch/issues/2

def runAction(sudoku, puzzl_mat, guess_mat, curs_pos, action:int, action_val:int): 
	
	# run the action, update the world, return the reward.
	# act = b % 4
	reward = -0.05
	if action == Action.UP.value : 
		curs_pos[0] -= 1
	if action == Action.RIGHT.value: 
		curs_pos[1] += 1
	if action == Action.DOWN.value: 
		curs_pos[0] += 1
	if action == Action.LEFT.value:
		curs_pos[1] -= 1
	# clip (rather than wrap) cursor position
	for i in range(2): 
		if curs_pos[i] < 0: 
			reward = -0.5
			curs_pos[i] = 0
		if curs_pos[i] >= SuN: 
			reward = -0.5
			curs_pos[i] = SuN - 1
		
	# if we're on a open cell, but no moves are possible, 
	# negative reward! 
	clue = puzzl_mat[curs_pos[0], curs_pos[1]]
	curr = guess_mat[curs_pos[0], curs_pos[1]]
	sudoku.setMat(puzzl_mat + guess_mat) # so that checkIfSafe works properly.
	if clue == 0 and curr == 0: 
		if not sudoku.checkOpen(curs_pos[0], curs_pos[1]): 
			reward = -1
	
	if action == Action.SET_GUESS.value:
		if clue == 0 and curr == 0 and sudoku.checkIfSafe(curs_pos[0], curs_pos[1], action_val):
			# updateNotes(curs_pos, action_val, notes)
			reward = 1
			guess_mat[curs_pos[0], curs_pos[1]] = action_val
		else:
			reward = -1
	if action == Action.UNSET_GUESS.value:
		if curr != 0: 
			guess_mat[curs_pos[0], curs_pos[1]] = 0
			reward = -1 # must exactly cancel, o/w best strategy is to simply set/unset guess repeatedly.
		else:
			reward = -1.25
			
	if False: 
		print(f'runAction @ {curs_pos[0]},{curs_pos[1]}: {action}:{action_val}')
	
	return reward

	
def encodeBoard(sudoku, puzzl_mat, guess_mat, curs_pos, action, action_val):  
	'''
	Encodes the current board state and encodes the given action,
		runs the action, and then encodes the new board state.
		Also returns a mask matrix (#nodes by #nodes) which represents parent/child relationships
		which defines the attention mask used in the transformer heads

	The board and action nodes have the same encoding- contains one hot of node type and node value
	
	Returns:
	board encoding: Shape (#board nodes x world_dim)
	action encoding: Shape (#action nodes x world_dim)
	new board encoding: Shape (#newboard nodes x world_dim)
	'''
	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action, action_val, 0.0)
	benc,coo,a2a = sparse_encoding.encodeNodes(nodes)
	
	reward = runAction(sudoku, puzzl_mat, guess_mat, curs_pos, action, action_val)
	
	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action, action_val, reward) # action_val doesn't matter
	newbenc,coo,a2a = sparse_encoding.encodeNodes(nodes)
	
	return benc, newbenc, coo, a2a, reward, reward_loc
	
def encode1DBoard():  
	# simple 1-D version of sudoku. 
	puzzle = np.arange(1, 10)
	mask = np.random.randint(0,3,9)
	puzzle = puzzle * (mask > 0)
	curs_pos = np.random.randint(0,9)
	action = 4
	action_val = np.random.randint(0,9)
	guess_mat = np.zeros((9,))

	nodes, reward_loc,_ = sparse_encoding.sudoku1DToNodes(puzzle, guess_mat, curs_pos, action, action_val, 0.0)
	benc,coo,a2a = sparse_encoding.encodeNodes(nodes)
	
	# run the action. 
	reward = -1
	if puzzle[action_val-1] == 0 and puzzle[curs_pos] == 0: 
		guess_mat[curs_pos] = action_val
		reward = 1
	
	nodes, reward_loc,_ = sparse_encoding.sudoku1DToNodes(puzzle, guess_mat, curs_pos, action, action_val, reward) # action_val doesn't matter
	newbenc,coo,a2a = sparse_encoding.encodeNodes(nodes)
	
	return benc, newbenc, coo, a2a, reward, reward_loc

	
def enumerateActionList(n:int): 
	action_types = []
	action_values = []
	# directions
	for at in [0,1,2,3]: 
		action_types.append(at)
		action_values.append(0)
	at = Action.SET_GUESS.value
	for av in range(SuN):
		action_types.append(at)
		action_values.append(av+1)
	# unset guess action
	action_types.append( Action.UNSET_GUESS.value )
	action_values.append( 0 )
	
	nactions = len(action_types)
	if len(action_types) < n: 
		rep = n // len(action_types) + 1
		action_types = action_types * rep
		action_values = action_values * rep
	if len(action_types) > n: 
		action_types = action_types[:n]
		action_values = action_values[:n]
	return action_types,action_values
	
def sampleActionList(n:int): 
	# this is slow but whatever, only needs to run once
	action_types = []
	action_values = []
	possible_actions = [ 0,1,2,3,4,4,4,4,4,5,5 ] # FIXME
	for i in range(n): 
		action = possible_actions[np.random.randint(len(possible_actions))]
		actval = 0
		if action == Action.SET_GUESS.value: 
			actval = np.random.randint(1,10)
		action_types.append(action)
		action_values.append(actval)
	
	return action_types,action_values


def enumerateBoards(puzzles, n, possible_actions=[], min_dist=1, max_dist=1): 
	'''
	Parameters:
	n: (int) Number of samples to generate
	min_dist: (int) Represents the min distance travelled.
	max_dist: (int) Represents the max distance travelled (inclusive)

	Returns:
	orig_board_enc: (tensor) Shape (N x #board nodes x 20), all the initial board encodings
	new_board_enc: (tensor) Shape (N x #board nodes x 20), all of the resulting board encodings due to actions
	rewards: (tensor) Shape (N,) Rewards of each episode 
	'''
	# changing the strategy: for each board, do all possible actions. 
	# this serves as a stronger set of constraints than random enumeration.
	try: 
		orig_board_enc = torch.load(f'orig_board_enc_{n}.pt')
		new_board_enc = torch.load(f'new_board_enc_{n}.pt')
		rewards_enc = torch.load(f'rewards_enc_{n}.pt')
		# need to get the coo, a2a, etc variables - so run one encoding.
		n = 1
	except Exception as error:
		print(colored(f"could not load precomputed data {error}", "red"))
		print("generating random board, action, board', reward")
	
	# action_types,action_values = enumerateActionList(n)
	action_types,action_values = sampleActionList(n)
		
	sudoku = Sudoku(SuN, SuK)
	orig_boards = [] 
	new_boards = []
	actions = []
	rewards = torch.zeros(n, dtype=g_dtype)
	curs_pos_b = torch.randint(SuN, (n,2),dtype=int)
	
	# for half the boards, select only open positions. 
	for i in range( n // 2 ): 
		puzzl = puzzles[i, :, :]
		while puzzl[curs_pos_b[i,0], curs_pos_b[i,1]] > 0: 
			curs_pos_b[i,:] = torch.randint(SuN, (1,2),dtype=int)
	
	for i,(at,av) in enumerate(zip(action_types,action_values)):
		puzzl = puzzles[i, :, :].numpy()
		# move half the clues to guesses (on average)
		# to force generalization over both!
		mask = np.random.randint(0,2, (SuN,SuN)) == 1
		guess_mat = puzzl * mask
		puzzl_mat = puzzl * (1-mask)
		curs_pos = curs_pos_b[i, :] # see above.
		
		benc,newbenc,coo,a2a,reward,reward_loc = encodeBoard(sudoku, puzzl_mat, guess_mat, curs_pos, at, av )
		# benc,newbenc,coo,a2a,reward,reward_loc = encode1DBoard()
		orig_boards.append(benc)
		new_boards.append(newbenc)
		rewards[i] = reward
		if i % 1000 == 999:
			print(".", end = "", flush=True)
		
	if n > 1: 
		orig_board_enc = torch.stack(orig_boards)
		new_board_enc = torch.stack(new_boards)
		rewards_enc = rewards
		torch.save(orig_board_enc, f'orig_board_enc_{n}.pt')
		torch.save(new_board_enc, f'new_board_enc_{n}.pt')
		torch.save(rewards_enc, f'rewards_enc_{n}.pt')
		
	return orig_board_enc, new_board_enc, coo, a2a, rewards_enc, reward_loc

def trainValSplit(data_matrix: torch.Tensor, num_validate):
	'''
	Split data matrix into train and val data matrices
	data_matrix: (torch.tensor) Containing rows of data
	num_validate: (int) If provided, is the number of rows in the val matrix
	
	This is OK wrt constraints, as the split is non-stochastic in the order.
	'''
	num_samples = data_matrix.size(0)
	if num_samples <= 1:
		raise ValueError(f"data_matrix needs to be a tensor with more than 1 row")
	
	training_data = data_matrix[:-num_validate]
	eval_data = data_matrix[-num_validate:]
	return training_data, eval_data


class SudokuDataset(Dataset):
	'''
	Dataset where the ith element is a sample containing orig_board_enc,
		new_board_enc, action_enc, graph_mask, board_reward
	'''
	def __init__(self, orig_boards, new_boards, rewards):
		self.orig_boards = orig_boards
		self.new_boards = new_boards
		self.rewards = rewards 

	def __len__(self):
		return self.orig_boards.size(0)

	def __getitem__(self, idx):
		sample = {
			'orig_board' : self.orig_boards[idx],
			'new_board' : self.new_boards[idx], 
			'reward' : self.rewards[idx].item(),
		}
		return sample


def getDataLoaders(puzzles, num_samples, num_validate):
	'''
	Returns a pytorch train and test dataloader
	'''
	data_dict, coo, a2a, reward_loc = getDataDict(puzzles, num_samples, num_validate)
	train_dataset = SudokuDataset(data_dict['train_orig_board'],
											data_dict['train_new_board'], 
											data_dict['train_rewards'])

	test_dataset = SudokuDataset(data_dict['test_orig_board'],
										data_dict['test_new_board'], 
										data_dict['test_rewards'])

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	return train_dataloader, test_dataloader, coo, a2a, reward_loc


def getDataDict(puzzles, num_samples, num_validate):
	'''
	Returns a dictionary containing training and test data
	'''
	orig_board, new_board, coo, a2a, rewards, reward_loc = enumerateBoards(puzzles, num_samples)
	print(orig_board.shape, new_board.shape, rewards.shape)
	train_orig_board, test_orig_board = trainValSplit(orig_board, num_validate)
	train_new_board, test_new_board = trainValSplit(new_board, num_validate)
	train_rewards, test_rewards = trainValSplit(rewards, num_validate)

	dataDict = {
		'train_orig_board': train_orig_board,
		'train_new_board': train_new_board,
		'train_rewards': train_rewards,
		'test_orig_board': test_orig_board,
		'test_new_board': test_new_board,
		'test_rewards': test_rewards
	}
	return dataDict, coo, a2a, reward_loc

def getMemoryDict():
	fd_board = make_mmf("board.mmap", [batch_size, token_cnt, world_dim])
	fd_new_board = make_mmf("new_board.mmap", [batch_size, token_cnt, world_dim])
	fd_boardp = make_mmf("boardp.mmap", [batch_size, token_cnt, world_dim])
	fd_reward = make_mmf("reward.mmap", [batch_size, reward_dim])
	fd_rewardp = make_mmf("rewardp.mmap", [batch_size, reward_dim])
	fd_attention = make_mmf("attention.mmap", [2, token_cnt, token_cnt, n_heads])
	fd_wqkv = make_mmf("wqkv.mmap", [n_heads*2,2*xfrmr_dim,xfrmr_dim])
	memory_dict = {'fd_board':fd_board, 'fd_new_board':fd_new_board, 'fd_boardp':fd_boardp,
					 'fd_reward': fd_reward, 'fd_rewardp': fd_rewardp, 'fd_attention': fd_attention,
					  'fd_wqkv':fd_wqkv }
	return memory_dict


def getLossMask(board_enc, device):
	'''
	mask off extra space for passing info between layers
	'''
	loss_mask = torch.ones(1, board_enc.shape[1], board_enc.shape[2], device=device,dtype=g_dtype)
	for i in range(11,20):
		loss_mask[:,:,i] *= 0.001 # semi-ignore the "latents"
	return loss_mask 

def getOptimizer(optimizer_name, model, lr=2.5e-4, weight_decay=0):
	if optimizer_name == "adam": 
		optimizer = optim.Adam(model.parameters(), lr=lr)
	elif optimizer_name == 'adamw':
		optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
	elif optimizer_name == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=lr*1e-3)
	else: 
		optimizer = psgd.LRA(model.parameters(),lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,\
			preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=20, grad_clip_max_norm=5.0)
	return optimizer 

def updateMemory(memory_dict, pred_dict): 
	'''
	Updates memory map with predictions.

	Args:
	memory_dict (dict): Dictionary containing memory map file descriptors.
	pred_dict (dict): Dictionary containing predictions.

	Returns:
	None
	'''
	write_mmap(memory_dict['fd_board'], pred_dict['old_board'].cpu())
	write_mmap(memory_dict['fd_new_board'], pred_dict['new_board'].cpu())
	write_mmap(memory_dict['fd_boardp'], pred_dict['new_state_preds'].cpu().detach())
	write_mmap(memory_dict['fd_reward'], pred_dict['rewards'].cpu())
	write_mmap(memory_dict['fd_rewardp'], pred_dict['reward_preds'].cpu().detach())
	if 'a1' in pred_dict and 'a2' in pred_dict:
		if (pred_dict['a1'] is not None) and (pred_dict['a2'] is not None):
			write_mmap(memory_dict['fd_attention'], torch.stack((pred_dict['a1'], pred_dict['a2']), 0))
	if (pred_dict['w1'] is not None) and (pred_dict['w2'] is not None):
		write_mmap(memory_dict['fd_wqkv'], torch.stack((pred_dict['w1'], pred_dict['w2']), 0))
	return 

def train(args, memory_dict, model, train_loader, optimizer, hcoo, reward_loc, uu):
	# model.train()
	sum_batch_loss = 0.0

	for batch_idx, batch_data in enumerate(train_loader):
		old_board, new_board, rewards = [t.to(args["device"]) for t in batch_data.values()]
		
		# scale down the highlight, see if the model can learn.. 
		# uu_scl = 1.0 - uu / args["NUM_ITERS"]
		# uu_scl = uu_scl / 10
		# old_board[:,:,Axes.H_AX.value] = old_board[:,:,Axes.H_AX.value] * uu_scl
		# new_board[:,:,Axes.H_AX.value] = new_board[:,:,Axes.H_AX.value] * uu_scl
		# appears much harder! 

		pred_data = {}
		if optimizer_name != 'psgd': 
			optimizer.zero_grad()
			new_state_preds,w1,w2 = \
				model.forward(old_board, hcoo, uu, None)
			reward_preds = new_state_preds[:,reward_loc, 32+26]
			pred_data = {'old_board':old_board, 'new_board':new_board, 'new_state_preds':new_state_preds,
					  		'rewards': rewards*5, 'reward_preds': reward_preds,
							'w1':w1, 'w2':w2}
			# new_state_preds dimensions bs,t,w 
			loss = torch.sum((new_state_preds[:,:,33:64] - new_board[:,:,1:32])**2)
			# adam is unstable -- attempt to stabilize?
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
			loss.backward()
			optimizer.step() 
			print(loss.detach().cpu().item())
		else: 
			# psgd library internally does loss.backwards and zero grad
			def closure():
				nonlocal pred_data
				new_state_preds,w1,w2 = model.forward(old_board, hcoo, uu, None)
				reward_preds = new_state_preds[:,reward_loc, 32+26]
				pred_data = {'old_board':old_board, 'new_board':new_board, 'new_state_preds':new_state_preds,
								'rewards': rewards*5, 'reward_preds': reward_preds,
								'w1':w1, 'w2':w2}
				loss = torch.sum((new_state_preds[:,:,33:64] - new_board[:,:,1:32])**2) + \
					sum( \
					[torch.sum(1e-4 * torch.rand_like(param,dtype=g_dtype) * param * param) for param in model.parameters()])
					# this was recommended by the psgd authors to break symmetries w a L2 norm on the weights. 
				return loss
			loss = optimizer.step(closure)
		
		lloss = loss.detach().cpu().item()
		print(lloss)
		args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
		args["fd_losslog"].flush()
		uu = uu + 1
		
		if uu % 1000 == 999: 
			model.save_checkpoint(f"checkpoints/racoonizer_{uu//1000}.pth")

		sum_batch_loss += lloss
		if batch_idx % 25 == 0:
			updateMemory(memory_dict, pred_data)
			pass 
	
	# add epoch loss
	avg_batch_loss = sum_batch_loss / len(train_loader)
	return uu
	
	
def validate(args, model, test_loader, optimzer_name, hcoo, uu):
	model.eval()
	sum_batch_loss = 0.0
	with torch.no_grad():
		for batch_data in test_loader:
			old_board, new_board, rewards = [t.to(args["device"]) for t in batch_data.values()]
			new_state_preds,w1,w2 = model.forward(old_board, hcoo, uu, None)
			reward_preds = new_state_preds[:,reward_loc, 32+26]
			loss = torch.sum((new_state_preds[:,:,33:64] - new_board[:,:,1:32])**2)
			lloss = loss.detach().cpu().item()
			print(f'v{lloss}')
			args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
			args["fd_losslog"].flush()
			sum_batch_loss += loss.cpu().item()
	
	avg_batch_loss = sum_batch_loss / len(test_loader)
			
	return 
	
def trainQfun(rollouts_board, rollouts_reward, rollouts_action, nn, memory_dict, model, qfun, hcoo, reward_loc, locs, name):
	n_roll = rollouts_board.shape[0]
	n_tok = rollouts_board.shape[1]
	width = rollouts_board.shape[2]
	pred_data = {}
	for uu in range(nn): 
		indx = torch.randint(0,n_roll,(batch_size,))
		boards = rollouts_board[indx,:,:].squeeze().float()
		reward = rollouts_reward[indx].squeeze()
		actions = rollouts_action[indx,:].squeeze()
		
		# expand boards
		boards = torch.cat((boards, torch.zeros(batch_size, n_tok, width)), dim=2)
		# remove the layer encoding; will be replaced in the model.
		boards[:,:,0] = 0
		boards = torch.round(boards * 4.0) / 4.0
		
		# sparse_encoding.decodeNodes("", boards[0,:,:].squeeze().float(), locs)
		# do we even need to encode the action? 
		# it should not have changed!  
		# for j in range(batch_size): 
		# 	at = actions[j,0]
		# 	av = actions[j,1]
		# 	aenc = sparse_encoding.encodeActionNodes(at, av)
		# 	s = aenc.shape[0]
		# 	print(torch.sum((boards[j,0:s,:] - aenc)**2).item())
			
		boards = boards.cuda()
		reward = reward.cuda()
		with torch.no_grad(): 
			model_boards,_,_ = model.forward(boards,hcoo,0,None)
			
		def closure(): 
			nonlocal pred_data
			qfun_boards,_,_ = qfun.forward(model_boards,hcoo,0,None)
			reward_preds = qfun_boards[:,reward_loc, 32+26]
			pred_data = {'old_board':boards, 'new_board':model_boards, 'new_state_preds':qfun_boards,
								'rewards': reward, 'reward_preds': reward_preds,
								'w1':None, 'w2':None}
			loss = torch.sum((reward - reward_preds)**2) + \
				sum([torch.sum(1e-4 * torch.rand_like(param,dtype=g_dtype) * param * param) for param in qfun.parameters()])
			return loss
		
		loss = optimizer.step(closure)
		lloss = loss.detach().cpu().item()
		print(lloss)
		args["fd_losslog"].write(f'{uu}\t{lloss}\n')
		args["fd_losslog"].flush()
		if uu % 25 == 0:
			updateMemory(memory_dict, pred_data)
			
		if uu % 1000 == 999: 
			qfun.save_checkpoint(f"checkpoints/{name}_{uu//1000}.pth")
		
	
class ANode: 
	def __init__(self, typ, val, reward, board_enc, index):
		self.action_type = typ
		self.action_value = val
		self.kids = []
		self.reward = reward
		# board_enc is the *result* of applying the action.
		self.board_enc = board_enc
		self.parent = None
		self.index = index
		
	def setParent(self, node): 
		self.parent = node
	
	def addKid(self, node): 
		self.kids.append(node)
		node.setParent(self)
		
	def getParent(self):
		return self.parent
		
	def getAltern(self): 
		res = []
		for k in self.kids: 
			if k.reward > 0.0: 
				res.append(k)
		return res
		
	def print(self, indent, all_actions=False): 
		color = "black"
		if self.reward > 0: 
			color = "green"
		if self.reward < -1.0: 
			color = "red"
		if self.action_type == 4 or all_actions: 
			print(colored(f"{indent}{self.action_type},{self.action_value},{self.reward} nkids:{len(self.kids)}", color))
			indent = indent + " "
		for k in self.kids: 
			k.print(indent, all_actions)

def plot_tensor(v, name, lo, hi):
	cmap_name = 'seismic'
	fig, axs = plt.subplots(1, 1, figsize=(12,6))
	data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
	data = np.reshape(data, (v.shape[0], v.shape[1]))
	im = axs.imshow(data, cmap = cmap_name)
	plt.colorbar(im, ax=axs)
	im.set_data(v)
	axs.set_title(name)
	axs.tick_params(bottom=True, top=True, left=True, right=True)
	plt.show()
	
def evaluateActions(model, mfun, qfun, board, hcoo, depth, reward_loc, locs, time, sum_contradiction, action_nodes):
	# evaluate all the possible actions for a current board
	# by running the forward transition model
	# this will also predict reward per action
	# clean up the boards
	board = np.round(board * 4.0) / 4.0
	board = torch.tensor(board).cuda()
	bs = board.shape[0]
	ntok = board.shape[1]
	width = board.shape[2]
	# plot_tensor(board.detach().cpu().numpy().T, "board", -4.0, 4.0)
		
	action_types,action_values = enumerateActionList(9+4+1)
	nact = len(action_types)
	
	# check if we could guess here = cursor loc has no clue or guess
	# (the model predicts the can_guess flag)
	board_loc,cursor_loc = locs
	can_guess = torch.clip(board[:,cursor_loc,26+4].clone().squeeze(),0,1)
	# puzzle is solved if all board_locs have either a clue or guess
	board_locf = torch.reshape(board_loc, (81,))
	clue_or_guess = torch.sum(board[:,board_locf, 11:20] > 0.5, dim=2)
	is_done = torch.prod(clue_or_guess, dim=1) > 0.5

	# make a batch with the new actions & replicated boards
	board = board.unsqueeze(1)
	new_boards = board.repeat(1, nact, 1, 1 )
	for i,(at,av) in enumerate(zip(action_types,action_values)):
		aenc = sparse_encoding.encodeActionNodes(at, av) # only need one eval!
		s = aenc.shape[0]
		new_boards[:, i, 0:s, :] = aenc.cuda()

	new_boards = new_boards.reshape(bs * nact, ntok, width)
	boards_pred,_,_ = model.forward(new_boards,hcoo,0,None)
	mfun_pred,_,_ = mfun.forward(boards_pred,None,0,None)
	qfun_pred,_,_ = qfun.forward(boards_pred,None,0,None)

	boards_pred = boards_pred.detach().reshape(bs, nact, ntok, width)
	mfun_pred = mfun_pred.detach().reshape(bs, nact, ntok, width)
	mfun_pred = mfun_pred[:,:,reward_loc, 32+26].clone().squeeze()
	qfun_pred = qfun_pred.detach().reshape(bs, nact, ntok, width)
	qfun_pred = qfun_pred[:,:,reward_loc, 32+26].clone().squeeze()
	reward_pred = boards_pred[:, :,reward_loc, 32+26].clone().squeeze()
	# copy over the beginning structure - needed!
	# this will have to be changed for longer-term memory TODO
	boards_pred[:,:,:,1:32] = boards_pred[:,:,:,33:]
	boards_pred[:,:,:,32:] = 0 # don't mess up forward computation
	boards_pred[:,:,reward_loc, 26] = 0 # it's a resnet - reset the reward.

	mask = reward_pred.clone() # torch.clip(reward_pred + 0.8, 0.0, 10.0)
		# reward-weighted; if you can guess, generally do that.
	valid_guesses = reward_pred[:,4:13] > 0
	mask[:,0:4] = mfun_pred[:,0:4] * 2
	pdb.set_trace()
	mask = F.softmax(mask, 1)
	# mask = mask / torch.sum(mask, 1).unsqueeze(1).expand(-1,14)
	indx = torch.multinomial(mask, 1).squeeze()

	lin = torch.arange(0,bs)
	boards_pred_taken = boards_pred[lin,indx,:,:].detach().squeeze().cpu().numpy()
	contradiction = can_guess * \
		(1-torch.clip(torch.sum(reward_pred[:,4:13] > 0, 1), 0, 1))
	# detect a contradiction when the square is empty = can_guess is True,
	# but there are no actions expected to result in reward.
	# would be nice if maybe this was not hard-coded..
	reward_pred_taken = reward_pred[lin,indx]
	reward_pred_taken = reward_pred_taken - 100*contradiction # end of game!
	reward_np = reward_pred_taken
	
	action_node_new = []
	indent = " " # * depth
	for j in range(bs):
		at_t = action_types[indx[j]]
		av_t = action_values[indx[j]]
		an_taken = ANode(at_t, av_t, reward_np[j], boards_pred_taken[j,:,:32].squeeze(), time)
		action_nodes[j].addKid(an_taken)
		action_node_new.append(an_taken) # return value of new nodes.
		if indx[j] >= 4 and indx[j] < 13: # was a guess (!contradiction)
			valid_guesses[j,indx[j]-4] = False # remove this option
			for m in range(9): 
				if valid_guesses[j,m]: 
					# store the node & its board encoding,
					#   for later backtracking
					# offset 4 is to skip the move actions.
					at = action_types[m+4]
					av = action_values[m+4]
					an = ANode(at, av, reward_pred[j,m+4], boards_pred[j,m+4,:,:32].cpu().numpy().squeeze(), time)
					action_nodes[j].addKid(an)
		if j == 0: 
			print(f"{time} action {getActionName(at_t)} {av_t} reward {reward_np[0].item()}")
			if contradicton[0] > 0.5:
				color = "red"
			else:
				color = "black"
			print(colored(f"contradiction {contradiction[0].item()}", color), end=" ")
			print(f"sum_contra {sum_contradiction[0].item()}", end=" ")
			if is_done[0]:
				color = "green"
			else:
				color = "black"
			print(colored(f"is_done {is_done[0]}", color))
			sparse_encoding.decodeNodes(indent, boards_pred_taken[0,:,:], locs)
		
	return boards_pred_taken, action_node_new, contradiction, is_done

	
def evaluateActionsBacktrack(model, mfun, qfun, puzzles, hcoo, nn):
	bs = 96
	pi = np.random.randint(0, puzzles.shape[0], (nn,bs))
	anode_list = []
	sudoku = Sudoku(SuN, SuK)
	for n in range(nn):
		puzzl_mat = np.zeros((bs,SuN,SuN))
		for k in range(bs):
			puzzl_mat[k,:,:] = puzzles[pi[n,k],:,:].numpy()
		guess_mat = np.zeros((bs,SuN,SuN))
		curs_pos = torch.randint(SuN, (bs,2),dtype=int)
		board = torch.zeros(bs,token_cnt,world_dim)
		for k in range(bs):
			nodes,reward_loc,locs = sparse_encoding.sudokuToNodes(puzzl_mat[k,:,:], guess_mat[k,:,:], curs_pos[k,:], 0, 0, 0.0) # action will be replaced.
			board[k,:,:],_,_ = sparse_encoding.encodeNodes(nodes)
		k = 0
		print(f"-- initial state {k}--")
		print(colored("-----", "green"))
		
		rollouts_board = torch.zeros(duration, bs, token_cnt, 32, dtype=torch.float16)
		rollouts_reward = torch.zeros(duration, bs)
		rollouts_action = torch.zeros(duration, bs, 2, dtype=int)
		rollouts_parent = torch.zeros(duration, bs, dtype=int)
		rollouts_board[0,:,:,:] = board[:,:,:32]
		board = board.numpy()
		sum_contradiction = torch.zeros(bs)
		# setup the root action nodes.
		root_nodes = [ANode(8,0,0.0,board[k,:,:32],0) for k in range(bs)]
		action_nodes = [root_nodes[k] for k in range(bs)]
		# since the reward will be updated, keep a ref list of nodes
		rollout_nodes = [[None for k in range(bs)] for _ in range(duration)]
		for time in range(duration-1): 
			with torch.no_grad(): 
				board_new, action_node_new, contradiction, is_done = evaluateActions(model, mfun, qfun, board, hcoo, 0, reward_loc,locs, time, sum_contradiction, action_nodes)
				sum_contradiction = sum_contradiction + contradiction.cpu()
			board = board_new
			# root_nodes[0].print("")
			# backtracking!
			for k in range(bs):
				# default: replace with new node
				action_nodes[k] = action_node_new[k]
				if contradiction[k] > 0: 
					an = action_node_new[k] # there will never be alternatives here
					m = time+1 # alternatives are added to the parents
					altern = an.getAltern()
					while m >= 0 and len(altern) < 1: 
						an.reward = -100 # propagate the contradiction back. 
						an = an.getParent()
						if an is None: 
							pdb.set_trace()
						m = m-1
						altern = an.getAltern()
					if len(altern) == 0:
						pdb.set_trace() # puzzle must be solvable!
					# probably should sort by reward here. meh, take the first one.
					action_nodes[k] = altern[0]
					board[k,:,:32] = altern[0].board_enc
					board[k,:,32:] = 0.0 # jic
					# if k == 0:
					# 	print(colored(f"[{k}] backtracking to {m+1}", "blue"))
					# 	action_nodes[k].print("")
				rollout_nodes[time+1][k] = action_nodes[k]

		for j in range(duration-1): 
			for k in range(bs):
				rollouts_board[j+1,k,:,:] = torch.tensor(rollout_nodes[j+1][k].board_enc, dtype=torch.float16)
				rollouts_reward[j+1,k] = rollout_nodes[j+1][k].reward
				rollouts_action[j+1,k, 0] = rollout_nodes[j+1][k].action_type
				rollouts_action[j+1,k, 1] = rollout_nodes[j+1][k].action_value
				rollouts_parent[j+1,k] = rollout_nodes[j+1][k].getParent().index

		# in these files, the board is the state resulting from the action.
		# likewise for reward, which is updated through rollouts.
		# parent indexes the board prior the action.
		torch.save(rollouts_board, f'rollouts/rollouts_board_{n}.pt')
		torch.save(rollouts_reward, f'rollouts/rollouts_reward_{n}.pt')
		torch.save(rollouts_action, f'rollouts/rollouts_action_{n}.pt')
		torch.save(rollouts_parent, f'rollouts/rollouts_parent_{n}.pt')

def moveValueDataset(puzzles, hcoo, bs, nn):
	# calculate the value of each square
	# as distance to closest empty square
	# then calculate move value as the derivative of this.

	try:
		boards = torch.load(f'rollouts/move_boards.pt')
		actions = torch.load(f'rollouts/move_actions.pt')
		rewards = torch.load(f'rollouts/move_rewards.pt')
		nn = 0
	except Exception as error:
		print(colored(f"could not load precomputed data {error}", "red"))

	if nn > 0:
		pi = np.random.randint(0, puzzles.shape[0], (nn,bs))
		boards = torch.zeros(nn,bs,token_cnt,32)
		actions = torch.zeros(nn,bs,2)
		rewards = torch.zeros(nn,bs)

		filts = []
		for r in range(3,26,2): # 3, 5, 7, 9, 11, 13, 15, 17
			filt = torch.zeros(1,1,r,r)
			c = r // 2
			for i in range(r):
				for j in range(r):
					if abs(i-c) + abs(j-c) <= r//2:
						filt[0,0,i,j] = 1.0
			filts.append(filt)

		for n in range(nn):
			# make the movements hard: only a few empties.
			num_empty = np.random.randint(1,8, (bs,))
			# these puzzles are degenerate, but that's ok
			puzzl_mat = np.random.randint(1,10,(bs,SuN,SuN))
			for k in range(bs):
				ne = num_empty[k]
				indx = np.random.randint(0,9, (ne,2))
				lin = np.arange(0,ne)
				puzzl_mat[k,indx[lin,0],indx[lin,1]] = 0
			# puzzl_mat = np.zeros((bs,SuN,SuN))
			# for k in range(bs):
			# 	puzzl_mat[k,:,:] = puzzles[pi[n,k],:,:].numpy()
			guess_mat = np.zeros((bs,SuN,SuN)) # should not matter..
			curs_pos = torch.randint(SuN, (bs,2),dtype=int)
			empty = torch.tensor(puzzl_mat[:,:,:] == 0, dtype=torch.float32)
			value_mat = torch.zeros(puzzl_mat.shape, dtype=torch.float32)
			value_mat = value_mat + 1.0 * empty
			value = torch.zeros(puzzl_mat.shape, dtype=torch.float32)
			value = value + value_mat
			for filt in filts:
				vf = torch.nn.functional.conv2d(value_mat.unsqueeze(1), filt, padding='same')
				vf = vf > 0.5
				value = value + vf.squeeze()
			# print(puzzl_mat[0,:,:].squeeze())
			# plt.imshow(value[0,:,:].squeeze().numpy())
			# plt.colorbar()
			# plt.show()
			# select a move, calculate value. cursor pos is [x,y]
			# x is hence row or up/down
			move = np.random.randint(0,4,(bs,))
			xnoty = move % 2 == 0
			direct = (move // 2) * 2 - 1
			direct = direct * (xnoty*2-1)
			new_curs = torch.zeros_like(curs_pos)
			new_curs[:,0] = curs_pos[:,0] + xnoty * direct
			new_curs[:,1] = curs_pos[:,1] + (1-xnoty) * direct
			hit_edge = (new_curs[:,0] < 0) + (new_curs[:,0] > 8) + \
				(new_curs[:,1] < 0) + (new_curs[:,1] > 8)
			new_curs = torch.clip(new_curs, 0, 8)
			lin = torch.arange(0,bs)
			orig_val = value[lin,curs_pos[:,0],curs_pos[:,1]]
			new_val = value[lin,new_curs[:,0],new_curs[:,1]]
			reward = new_val - orig_val
			reward = reward * (~hit_edge)

			for k in range(bs):
				nodes,reward_loc,locs = sparse_encoding.sudokuToNodes(puzzl_mat[k,:,:], guess_mat[k,:,:], curs_pos[k,:], move[k], 0, 0)
				board,_,_ = sparse_encoding.encodeNodes(nodes)
				boards[n,k,:,:] = board[:,:32]
			actions[n,:,0] = torch.tensor(move)
			rewards[n,:] = reward
			if n % 5 == 4:
				print(".", end = "", flush=True)

		torch.save(boards, 'rollouts/move_boards.pt')
		torch.save(actions, 'rollouts/move_actions.pt')
		torch.save(rewards, 'rollouts/move_rewards.pt')

	return boards,actions,rewards

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train sudoku world model")
	parser.add_argument('-c', action='store_true', help='clear, start fresh for training world model from random rollouts')
	parser.add_argument('-e', action='store_true', help='evaluate with backtracking')
	parser.add_argument('-t', action='store_true', help='train')
	parser.add_argument('-q', action='store_true', help='train Q function')
	parser.add_argument('-m', action='store_true', help='train Q function for movements')
	parser.add_argument('-r', type=int, default=1, help='rollout file number')
	cmd_args = parser.parse_args()
	
	puzzles = torch.load(f'puzzles_{SuN}_500000.pt')
	NUM_TRAIN = batch_size * 1800 
	NUM_VALIDATE = batch_size * 300
	NUM_SAMPLES = NUM_TRAIN + NUM_VALIDATE
	NUM_ITERS = 100000
	device = torch.device('cuda:0') 
	# can override with export CUDA_VISIBLE_DEVICES=1 
	torch.set_float32_matmul_precision('high')
	args = {"NUM_TRAIN": NUM_TRAIN, "NUM_VALIDATE": NUM_VALIDATE, "NUM_SAMPLES": NUM_SAMPLES, "NUM_ITERS": NUM_ITERS, "device": device}
	
	# get our train and test dataloaders
	train_dataloader, test_dataloader, coo, a2a, reward_loc = getDataLoaders(puzzles, args["NUM_SAMPLES"], args["NUM_VALIDATE"])
	# print(reward_loc)
	# print(coo)
	
	# first half of heads are kids to parents
	kids2parents, dst_mxlen_k2p, _ = expandCoo(coo)
	# swap dst and src
	coo_ = torch.zeros_like(coo) # type int32: indexes
	coo_[:,0] = coo[:,1]
	coo_[:,1] = coo[:,0]
	parents2kids, dst_mxlen_p2k, _ = expandCoo(coo_)
	# and self attention (intra-token attention ops) -- either this or add a second MLP layer. 
	coo_ = torch.arange(token_cnt).unsqueeze(-1).tile([1,2])
	self2self, dst_mxlen_s2s, _ = expandCoo(coo_)
	# add global attention
	all2all = torch.Tensor(a2a); 
	# coo_ = torch.zeros((token_cnt**2-token_cnt, 2), dtype=int)
	# k = 0
	# for i in range(token_cnt): 
	# 	for j in range(token_cnt): 
	# 		if i != j: # don't add the diagonal.
	# 			coo_[k,:] = torch.tensor([i,j],dtype=int)
	# 			k = k+1
	# all2all, dst_mxlen_a2a, _ = expandCoo(coo_)
	kids2parents = kids2parents.cuda()
	parents2kids = parents2kids.cuda()
	self2self = self2self.cuda()	
	all2all = all2all.cuda()
	hcoo = [(kids2parents,dst_mxlen_k2p), (parents2kids,dst_mxlen_p2k), \
		(self2self, dst_mxlen_s2s), all2all]
	
	# allocate memory
	memory_dict = getMemoryDict()
	
	# define model 
	model = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=n_heads, n_layers=8, repeat=3, mode=0).to(device)
	model.printParamCount()
	
	# movement predictor
	mfun = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=4, n_layers=2, repeat=2, mode=0).to(device)
	mfun.printParamCount()

	# qfun predictor
	qfun = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=n_heads, n_layers=8, repeat=3, mode=0).to(device)
	qfun.printParamCount()

	optimizer_name = "psgd" # adam, adamw, psgd, or sgd
	optimizer = getOptimizer(optimizer_name, model)

	if cmd_args.c: 
		print(colored("not loading any model weights.", "blue"))
	else:
		try:
			def getLatestFile(prefix):
				checkpoint_dir = 'checkpoints/'
				files = glob.glob(os.path.join(checkpoint_dir, prefix))
				# Filter out directories and only keep files
				files = [f for f in files if os.path.isfile(f)]
				if not files:
					raise ValueError("No models found in the checkpoint directory")
				# Find the most recently modified file
				latest_file = max(files, key=os.path.getmtime)
				print(colored(latest_file, "green"))
				return latest_file


			model.load_checkpoint(getLatestFile("rac*"))
			print(colored("loaded model checkpoint", "blue"))

			mfun.load_checkpoint(getLatestFile("mouse*"))
			print(colored("loaded mfun checkpoint", "blue"))

			qfun.load_checkpoint(getLatestFile("quail*"))
			print(colored("loaded qfun checkpoint", "blue"))
			time.sleep(1)
		except Exception as error:
			print(colored(f"could not load model checkpoint {error}", "red"))

	if cmd_args.e: 
		instance = cmd_args.r
		anode_list = evaluateActionsBacktrack(model, mfun, qfun, puzzles, hcoo, 319)
				
	if cmd_args.m:
		bs = 96
		nn = 1500
		rollouts_board,rollouts_action,rollouts_reward = moveValueDataset(puzzles, hcoo, bs,nn)

		optimizer = getOptimizer(optimizer_name, mfun)

		# get the locations of the board nodes.
		_,reward_loc,locs = sparse_encoding.sudokuToNodes(torch.zeros(9,9),torch.zeros(9,9),torch.zeros(2,dtype=int),0,0,0.0)

		rollouts_board = rollouts_board.reshape(nn*bs,token_cnt,32)
		rollouts_action = rollouts_action.reshape(nn*bs,2)
		rollouts_reward = rollouts_reward.reshape(nn*bs)

		fd_losslog = open('losslog.txt', 'w')
		args['fd_losslog'] = fd_losslog
		trainQfun(rollouts_board, rollouts_reward, rollouts_action, 200000, memory_dict, model, mfun, None, reward_loc, locs, "mouseizer")
		# note: no hcoo; only all-to-all attention

	if cmd_args.q: 
		bs = 96
		nfiles = 89
		rollouts_board = torch.zeros(duration, bs*nfiles, token_cnt, 32, dtype=torch.float16)
		# rollouts_parent_board = torch.zeros_like(rollouts_board)
		rollouts_reward = torch.zeros(duration, bs*nfiles)
		rollouts_action = torch.zeros(duration, bs*nfiles, 2, dtype=int)
		# rollouts_parent = torch.zeros(duration, bs*nfiles, dtype=int)
		
		for i in range(nfiles): 
			r_board = torch.load(f'rollouts/rollouts_board_{i}.pt')
			r_reward = torch.load(f'rollouts/rollouts_reward_{i}.pt')
			r_action = torch.load(f'rollouts/rollouts_action_{i}.pt')
			# r_parent = torch.load(f'rollouts/rollouts_parent_{i}.pt')
			rollouts_board[:,bs*i:bs*(i+1),:,:] = r_board
			rollouts_reward[:,bs*i:bs*(i+1)] = r_reward
			rollouts_action[:,bs*i:bs*(i+1),:] = r_action
			# for j in range(duration):
			# 	pdb.set_trace()
			# 	rollouts_parent_board[j,bs*i:bs*(i+1),:,:] = r_board[r_parent[j,:],:,:,:] # this ought to work across bs
			print(f"loaded rollouts/board - reward - action {i} .pt")
		# plt.plot(rollouts_reward[:, 0:10].numpy())
		# plt.show()

		# flatten to get uniform actions
		rollouts_board = rollouts_board.reshape(duration*bs*nfiles, token_cnt, 32)
		# rollouts_parent_board = rollouts_parent_board.reshape(duration*bs*nfiles, token_cnt, 32)
		rollouts_reward = rollouts_reward.reshape(duration*bs*nfiles)
		rollouts_action = rollouts_action.reshape(duration*bs*nfiles, 2)

		# need to select only guess actions --
		# the moves are handled by mouseizer.
		guess_index = (rollouts_action[:,0] == 4).nonzero().squeeze()
		rollouts_board = rollouts_board[guess_index, :, :]
		# rollouts_parent_board = rollouts_parent_board[guess_index, :, :]
		rollouts_reward = rollouts_reward[guess_index]
		rollouts_reward = torch.clip(rollouts_reward, -15, 5)
		rollouts_action = rollouts_action[guess_index,:]

		optimizer = getOptimizer(optimizer_name, qfun)

		# get the locations of the board nodes.
		_,reward_loc,locs = sparse_encoding.sudokuToNodes(torch.zeros(9,9),torch.zeros(9,9),torch.zeros(2,dtype=int),0,0,0.0)

		fd_losslog = open('losslog.txt', 'w')
		args['fd_losslog'] = fd_losslog
		trainQfun(rollouts_board, rollouts_reward, rollouts_action, 200000, memory_dict, model, qfun, hcoo, reward_loc, locs, "quailizer")
		
	
	uu = 0
	if cmd_args.t:
		fd_losslog = open('losslog.txt', 'w')
		args['fd_losslog'] = fd_losslog
		while uu < NUM_ITERS:
			uu = train(args, memory_dict, model, train_dataloader, optimizer, hcoo, reward_loc, uu)
		
		# save after training
		model.save_checkpoint()
 
		# print("validation")
		validate(args, model, test_dataloader, optimizer_name, hcoo, uu)
