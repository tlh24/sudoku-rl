import math
import random
import argparse
import time
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pdb
from termcolor import colored
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

def runAction(sudoku, guess_mat, curs_pos, action:int, action_val:int): 
	
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
		
	# curs_pos[0] = curs_pos[0] % SuN # wrap at the edges; 
	# curs_pos[1] = curs_pos[1] % SuN # works for negative nums
	
	if action == Action.SET_GUESS.value:
		clue = sudoku.mat[curs_pos[0], curs_pos[1]]
		curr = guess_mat[curs_pos[0], curs_pos[1]]
		if clue == 0 and curr == 0 and sudoku.checkIfSafe(curs_pos[0], curs_pos[1], action_val):
			# updateNotes(curs_pos, action_val, notes)
			reward = 1
			guess_mat[curs_pos[0], curs_pos[1]] = action_val
		else:
			reward = -1
	if action == Action.UNSET_GUESS.value:
		curr = guess_mat[curs_pos[0], curs_pos[1]]
		if curr != 0: 
			guess_mat[curs_pos[0], curs_pos[1]] = 0
		else:
			reward = -0.5
			
	if False: 
		print(f'runAction @ {curs_pos[0]},{curs_pos[1]}: {action}:{action_val}')
	
	return reward

	
def encodeBoard(sudoku, guess_mat, curs_pos, action, action_val):  
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
	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, action, action_val, 0.0)
	benc,coo,a2a = sparse_encoding.encodeNodes(nodes)
	
	reward = runAction(sudoku, guess_mat, curs_pos, action, action_val)
	
	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, action, action_val, reward) # action_val doesn't matter
	newbenc,coo,a2a = sparse_encoding.encodeNodes(nodes)
	
	return benc, newbenc, coo, a2a, reward, reward_loc


# def generateActionValue(action: int, min_dist: int, max_dist: int):
# 	'''
# 	Generates an action value corresponding to the action.
# 	For movement actions, samples a dist unif on [min_dist, max_dist] and 
# 		chooses - or + direction based on the action (ex: -1 for left, +1 for right).
# 
# 	min_dist: (int) Represents the min distance travelled.
# 	max_dist: (int) Represents the max distance travelled.
# 	'''
# 	# movement action
# 	dist = np.random.randint(low=min_dist, high=max_dist+1)
# 	if action in [Action.DOWN.value, Action.LEFT.value]:
# 		direction = -1
# 		return dist * direction 
# 
# 	if action in [Action.UP.value, Action.RIGHT.value]:
# 		direction = 1
# 		return dist * direction 
# 
# 	# guess or set note action
# 	if action in [Action.SET_GUESS.value, Action.SET_NOTE.value]:
# 		return np.random.randint(1,SuN+1)
# 
# 	# nop
# 	return 0

	
def makeActionList(): 
	action_types = []
	action_values = []
	# # directions
	# for at in [0,1,2,3]: 
	# 	action_types.append(at)
	# 	action_values.append(0)
	at = Action.SET_GUESS.value
	for av in range(SuN):
		action_types.append(at)
		action_values.append(av+1)
	# # unset guess action
	# action_types.append( Action.UNSET_GUESS.value )
	# action_values.append( 0 )
	return action_types,action_values
	
def enumerateMoves(depth, episode, possible_actions=[]): 
	if not possible_actions:
		# possible_actions = [ 0,1,2,3 ]
		possible_actions = [ 0,1,2,3,4,4,4,4,4,5,5] # FIXME
		# possible_actions = [ 4,4,4,4 ]
		# possible_actions.append(Action.SET_GUESS.value) # upweight
		# possible_actions.append(Action.SET_GUESS.value)
	outlist = []
	if depth > 0: 
		for action in possible_actions:
			outlist.append(episode + [action])
			outlist = outlist + enumerateMoves(depth-1, episode + [action], possible_actions)
	return outlist


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
		
	action_types,action_values = makeActionList()
	nactions = len(action_types)
	if len(action_types) < n: 
		rep = n // len(action_types) + 1
		action_types = action_types * rep
		action_values = action_values * rep
	if len(action_types) > n: 
		action_types = action_types[:n]
		action_values = action_values[:n]
	sudoku = Sudoku(SuN, SuK)
	orig_boards = [] 
	new_boards = []
	actions = []
	rewards = torch.zeros(n, dtype=g_dtype)
	npuzzles = (n + nactions - 1) // nactions
	curs_pos_b = torch.randint(SuN, (npuzzles,2),dtype=int)
	
	# select only open positions. 
	for pi in range( npuzzles ): 
		puzzl = puzzles[pi, :, :]
		while puzzl[curs_pos_b[pi,0], curs_pos_b[pi,1]] > 0: 
			curs_pos_b[pi,:] = torch.randint(SuN, (1,2),dtype=int)
	
	for i,(at,av) in enumerate(zip(action_types,action_values)):
		pi = i // nactions
		puzzl = puzzles[pi, :, :]
		sudoku.setMat(puzzl.numpy())
		guess_mat = np.zeros((SuN, SuN))
		curs_pos = curs_pos_b[pi, :] # force constraints! 
		
		benc,newbenc,coo,a2a,reward,reward_loc = encodeBoard(sudoku, guess_mat, curs_pos, at, av )
		orig_boards.append(benc)
		new_boards.append(newbenc)
		rewards[i] = reward
		
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
					# torch.sum((new_state_preds[:,:,:20] - new_board[:,:,:20])**2)*1e-4 + \
					# torch.sum((new_state_preds[:,:,21:] - new_board[:,:,21:])**2)*1e-4 + \
					# we seem to have lost the comment explaining why this was here 
					# but it was recommended by the psgd authors to break symmetries w a L2 norm on the weights. 
				return loss
			loss = optimizer.step(closure)
		
		lloss = loss.detach().cpu().item()
		print(lloss)
		args["fd_losslog"].write(f'{uu}\t{lloss}\n')
		args["fd_losslog"].flush()
		uu = uu + 1
		
		if uu % 1000 == 999: 
			model.save_checkpoint(f"checkpoints/racoonizer_{uu//1000}.pth") #fixme

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
			fd_losslog.write(f'{uu}\t{lloss}\n')
			fd_losslog.flush()
			sum_batch_loss += loss.cpu().item()
	
	avg_batch_loss = sum_batch_loss / len(test_loader)
			
	return 

# def evaluateActions(model, sudoku, guess_mat, curs_pos, action_type, action_value, hcoo):
# 	# evaluate a batch of actions on the model, return the new predicted states.
# 	l = len(action_type)
# 	boards = torch.zeros(l, token_cnt, world_dim)
# 
# 	for i,(action,action_val) in enumerate(zip(action_type, action_value)):
# 		nodes,reward_loc,locs = sparse_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, action, action_val, 0.0)
# 		benc,coo = sparse_encoding.encodeNodes(nodes)
# 		boards[i,:,:] = benc
# 
# 	boards = boards.cuda()
# 	boards_pred,_,_,_,_ = model.forward(boards,hcoo,0,None)
# 	reward_pred = boards_pred[:,reward_loc, 20]
# 
# 	return boards_pred,reward_pred,locs
# 
# # seems slightly stupid to simulate the world through the model
# # when we have a perfectly good simulator at hand..
# # but, the point is to learn the dynamics,
# # and from that features of the world that are useful in predicting policy and value ..
# # (the original plan was to use value gradients, but Dreamer v2 v3 seems to indicate that does not work well with discrete domains -- instead, they use REINFORCE gradients.
# 
# 
# def evaluateActionsMany(model, sudoku, guess_mat, curs_pos, hcoo):
# 	action_types,action_values = makeActionList()
# 
# 	boards_pred, reward_pred, locs = evaluateActions(model, sudoku, guess_mat, curs_pos, action_types, action_values, hcoo)
# 
# 	for i,(at,av) in enumerate(zip(action_types,action_values)):
# 		reward = reward_pred[i]
# 		print(f"action type:{at} value:{av} reward:{reward}")
# 
# 	return boards_pred, reward_pred
# 
# # need some way of updating the actions without updating the whole board...
# 
# def evaluateActionsMany2(model, puzzles, hcoo):
# 	i = np.random.randint(puzzles.shape[0])
# 	puzzle = puzzles[i,:,:]
# 	sudoku = Sudoku(SuN, SuK)
# 	sudoku.setMat(puzzle.numpy())
# 	guess_mat = np.zeros((SuN, SuN))
# 	curs_pos = torch.randint(SuN, (2,),dtype=int)
# 
# 	boards_pred, reward_pred = evaluateActionsMany(model, sudoku, guess_mat, curs_pos, hcoo)
# 
# 	return boards_pred, reward_pred
	
class ANode: 
	def __init__(self, typ, val, reward): 
		self.action_type = typ
		self.action_value = val
		self.kids = []
		self.reward = reward
		
	def addKid(self, node): 
		self.kids.append(node)
		
	def print(self, indent): 
		print(indent, self.action_type, self.action_value, self.reward)
		indent = indent + " "
		for k in self.kids: 
			k.print(indent)

	
def evaluateActionsRec(model, board, hcoo, action_node, depth, locs): 
	# clean up the boards
	board = torch.round(board * 2.0) / 2.0
		
	action_types,action_values = makeActionList()

	# make a batch with the new actions & replicated boards
	new_boards = board.repeat(len(action_types), 1, 1 )
	for i,(at,av) in enumerate(zip(action_types,action_values)):
		aenc = sparse_encoding.encodeActionNodes(at, av)
		s = aenc.shape[0]
		new_boards[i, 0:s, :] = aenc.cuda()

	boards_pred,_,_ = model.forward(new_boards,hcoo,0,None)
	reward_pred = boards_pred[:,reward_loc, 26]
	# copy over the beginning structure - needed!
	# this will have to be changed! TODO
	boards_pred[:,:,33:] = 0

	for i,(at,av) in enumerate(zip(action_types,action_values)):
		reward = reward_pred[i].item()
		an = ANode(at, av, reward)
		action_node.addKid(an)
		indent = " " * depth
		print(f"{indent}action {getActionName(at)} {av} reward {reward}")
		sparse_encoding.decodeNodes(indent, boards_pred[i,:,:], locs)
		if reward > -1.0 and depth < 1:
			print(colored(f"{indent}->", "green"))
			evaluateActionsRec(model, boards_pred[i,:,:], hcoo, an, depth+1, locs)
		elif reward > -1.0:
			print(colored(f"{indent}-o", "green"))
		else:
			print(colored(f"{indent}-x", "yellow"))

	
def evaluateActionsRecurse(model, puzzles, hcoo): 
	i = np.random.randint(puzzles.shape[0])
	puzzle = puzzles[i,:,:]
	sudoku = Sudoku(SuN, SuK)
	sudoku.setMat(puzzle.numpy())
	guess_mat = np.zeros((SuN, SuN))
	curs_pos = torch.randint(SuN, (2,),dtype=int)
	print("-- initial state --")
	sudoku.printSudoku("", curs_pos, guess_mat)
	print(f"curs_pos {curs_pos[0]},{curs_pos[1]}")
	print(colored("-----", "green"))
	
	nodes,reward_loc,locs = sparse_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, 0, 0, 0.0) # action will be replaced.
	board,_,_ = sparse_encoding.encodeNodes(nodes)
	
	board = board.cuda()
	
	action_node = ANode(8,0,0.0) # root node
	evaluateActionsRec(model, board, hcoo, action_node, 0, locs)
	
	action_node.print("")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train sudoku world model")
	parser.add_argument('-c', action='store_true', help='start fresh')
	cmd_args = parser.parse_args()
	
	puzzles = torch.load(f'puzzles_{SuN}_500000.pt')
	NUM_TRAIN = batch_size * 1000 # 10 is too small
	NUM_VALIDATE = batch_size * 100
	NUM_SAMPLES = NUM_TRAIN + NUM_VALIDATE
	NUM_ITERS = 100000
	device = torch.device('cuda:0') 
	# can override with export CUDA_VISIBLE_DEVICES=1 
	torch.set_float32_matmul_precision('high')
	fd_losslog = open('losslog.txt', 'w')
	args = {"NUM_TRAIN": NUM_TRAIN, "NUM_VALIDATE": NUM_VALIDATE, "NUM_SAMPLES": NUM_SAMPLES, "NUM_ITERS": NUM_ITERS, "device": device, "fd_losslog": fd_losslog}
	
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
	model = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, reward_dim=1).to(device) 
	model.printParamCount()

	if cmd_args.c: 
		print(colored("not loading any model weights.", "blue"))
	else:
		try:
			checkpoint_dir = 'checkpoints/'
			files = os.listdir(checkpoint_dir)
			files = [os.path.join(checkpoint_dir, f) for f in files]
			# Filter out directories and only keep files
			files = [f for f in files if os.path.isfile(f)]
			if not files:
				raise ValueError("No files found in the checkpoint directory")
			# Find the most recently modified file
			latest_file = max(files, key=os.path.getmtime)
			print(colored(latest_file, "green"))
			model.load_checkpoint(latest_file)
			print(colored("loaded model checkpoint", "blue"))
			time.sleep(1)
		except Exception as error:
			print(colored(f"could not load model checkpoint {error}", "red"))
	
	optimizer_name = "psgd" # adam, adamw, psgd, or sgd
	optimizer = getOptimizer(optimizer_name, model)

	# evaluateActionsRecurse(model, puzzles, hcoo)

	uu = 0
	while uu < NUM_ITERS:
		uu = train(args, memory_dict, model, train_dataloader, optimizer, hcoo, reward_loc, uu)
	#
	# # save after training
	# model.save_checkpoint()
 
	print("validation")
	validate(args, model, test_dataloader, optimizer_name, hcoo, uu)
