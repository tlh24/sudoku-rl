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
# from netdenoise import NetDenoise
# from test_gtrans import getTestDataLoaders, SimpleMLP
from constants import *
# from tqdm import tqdm
import time 
import sys 
import argparse 
from type_file import Action, Axes, getActionName
from l1attn_sparse_cuda import expandCoo
import anode
import board_ops
# import psgd_20240912 as psgd
import psgd
	# https://sites.google.com/site/lixilinx/home/psgd
	# https://github.com/lixilinx/psgd_torch/issues/2

from utils import set_seed
# from diffusion.data import getBaselineDataloaders
# from diffusion.model import GPTConfig, GPT

# sys.path.insert(0, "baseline/")

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

def getDataDict(puzzles):
	'''
	Returns a dictionary containing training and test data
	'''
	orig_board, new_board, rewards = board_ops.enumerateBoards(puzzles)
	num_samples = orig_board.shape[0]
	num_validate = num_samples // 10
	# non-deterministic train / test split
	indx = torch.randperm(num_samples)
	def split(x):
		train = x[indx[:-num_validate]]
		test = x[indx[-num_validate:]]
		return train,test

	print(orig_board.shape, new_board.shape, rewards.shape)
	train_orig_board, test_orig_board = split(orig_board)
	train_new_board, test_new_board = split(new_board)
	train_rewards, test_rewards = split(rewards)

	dataDict = {
		'train_orig_board': train_orig_board,
		'train_new_board': train_new_board,
		'train_rewards': train_rewards,
		'test_orig_board': test_orig_board,
		'test_new_board': test_new_board,
		'test_rewards': test_rewards
	}
	return dataDict

def getDataLoaders(puzzles):
	'''
	Returns a pytorch train and test dataloader for gracoonizer world model training
	'''
	data_dict = getDataDict(puzzles)
	train_dataset = SudokuDataset(data_dict['train_orig_board'],
											data_dict['train_new_board'],
											data_dict['train_rewards'])

	test_dataset = SudokuDataset(data_dict['test_orig_board'],
										data_dict['test_new_board'],
										data_dict['test_rewards'])

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	return train_dataloader, test_dataloader

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


def getOptimizer(optimizer_name, model, lr=2.5e-4, weight_decay=0):
	if optimizer_name == "adam": 
		optimizer = optim.Adam(model.parameters(), lr=lr)
	elif optimizer_name == 'adamw':
		optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
	elif optimizer_name == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=lr*1e-3)
	else: 
		optimizer = psgd.LRA(model.parameters(),\
			lr_params=0.01,lr_preconditioner=0.02, momentum=0.9,\
			preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=20, grad_clip_max_norm=5.0)
		# grad clipping at 2 seems to slow things a bit
	return optimizer 


def updateMemory(memory_dict, pred_dict): 
	'''
	Updates mmap files for visualization.
	Args:
	memory_dict (dict): Dictionary containing memory map file descriptors.
	pred_dict (dict): Dictionary containing predictions.
	Returns nothing.
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


def train(args, memory_dict, model, train_loader, optimizer, hcoo, reward_loc, uu, inverse_wm=False):
	''' this trains the *world model* on random actions placed
	on random boards '''
	# model.train()
	for batch_indx, batch_data in enumerate(train_loader):
		if inverse_wm: 
			# transpose: predict the old board from the new board
			new_board, old_board, rewards = [t.to(args["device"]) for t in batch_data.values()]
		else: 
			old_board, new_board, rewards = [t.to(args["device"]) for t in batch_data.values()]
		
		# expand the boards to 64
		old_board = torch.cat((old_board, torch.zeros_like(old_board)), dim=-1).float()
		new_board = torch.cat((new_board, torch.zeros_like(new_board)), dim=-1).float()

		pred_data = {}
		if optimizer_name != 'psgd': 
			optimizer.zero_grad()
			new_state_preds = \
				model.forward(old_board, hcoo)
			reward_preds = new_state_preds[:,reward_loc, 32+Axes.R_AX.value]
			pred_data = {'old_board':old_board, 'new_board':new_board, 'new_state_preds':new_state_preds,
					  		'rewards': rewards*5, 'reward_preds': reward_preds,
							'w1':None, 'w2':None}
			# new_state_preds dimensions bs,t,w 
			loss = torch.sum((new_state_preds[:,:,33:64] - new_board[:,:,1:32])**2)
			# adam can be unstable -- attempt to stabilize?
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			loss.backward()
			optimizer.step() 
		else: 
			# psgd library internally does loss.backwards and zero grad
			def closure():
				nonlocal pred_data
				new_state_preds = model.forward(old_board, hcoo)
				reward_preds = new_state_preds[:,reward_loc, 32+Axes.R_AX.value]
				pred_data = {'old_board':old_board, 'new_board':new_board, 'new_state_preds':new_state_preds,
								'rewards': rewards*5, 'reward_preds': reward_preds,
								'w1':None, 'w2':None}
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
			if inverse_wm: 
				fname = "racoonizer_inv"
			else: 
				fname = "racoonizer"
			model.save_checkpoint(f"checkpoints/{fname}_{uu//1000}.pth")

		if batch_indx % 25 == 0:
			updateMemory(memory_dict, pred_data)
			pass 
	
	return uu
	
	
def validate(args, model, test_loader, optimzer_name, hcoo, uu, inverse_wm=False):
	model.eval()
	with torch.no_grad():
		for batch_indx, batch_data in enumerate(test_loader):
			if inverse_wm:
				# transpose: predict the old board from the new board
				new_board, old_board, rewards = [t.to(args["device"]) for t in batch_data.values()]
			else:
				old_board, new_board, rewards = [t.to(args["device"]) for t in batch_data.values()]

			# expand the boards to 64
			old_board = torch.cat((old_board, torch.zeros_like(old_board)), dim=-1).float()
			new_board = torch.cat((new_board, torch.zeros_like(new_board)), dim=-1).float()

			new_state_preds = model.forward(old_board, hcoo)
			reward_preds = new_state_preds[:,reward_loc, 32+26]
			loss = torch.sum((new_state_preds[:,:,33:64] - new_board[:,:,1:32])**2)
			lloss = loss.detach().cpu().item()
			print(f'v{lloss}')
			args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
			args["fd_losslog"].flush()
	return
			
	
def trainPolicy(rollouts_board, rollouts_reward, nn, memory_dict, model, qfun, hcoo, hcoo_m, reward_loc, locs, name):
	n_roll = rollouts_board.shape[0]
	n_tok = rollouts_board.shape[1]
	width = rollouts_board.shape[2]
	pred_data = {}

	# need to make a null board for getting the man-reward tokens.
	nodes,_,_ = sparse_encoding.sudokuToNodes(torch.zeros(9,9), torch.zeros(9,9), torch.zeros(2), 8, 0, many_reward=True)
	rw_boards,_,_ = sparse_encoding.encodeNodes(nodes)
		# get the coo and a2a vectors from the calling function, in hcoo_m
	reward_enc = rw_boards[-4:, :]
	reward_enc = torch.expand(reward_enc[None,:,:], batch_size)

	for uu in range(nn): 
		indx = torch.randint(0,n_roll,(batch_size,))
		boards = rollouts_board[indx,:,:].squeeze().float()
		reward = rollouts_reward[indx,:4].squeeze()
		
		# add on reward tokens.
		boards = torch.cat((boards, reward_enc), dim=1)
		# expand boards to 64 wide
		boards = torch.cat((boards, torch.zeros(batch_size, n_tok+4, width, device=boards.device)), dim=2)
		# remove the layer encoding; will be replaced in the model.
		boards[:,:,0] = 0
		boards = torch.round(boards * 4.0) / 4.0
			
		boards = boards.cuda()
		reward = reward.cuda()
		# with torch.no_grad():
			# model_boards = model.forward(boards,hcoo,0,None)
			# plt.plot((model_boards[:,reward_loc, 32+26] - reward).cpu().numpy())
			# pdb.set_trace()
		def closure(): 
			nonlocal pred_data
			qfun_boards = qfun.forward(boards,hcoo_m,0,None)
			with torch.no_grad(): 
				new_boards = boards.detach().clone()
				# new_boards[:,0:3, 20:24] = torch.tile(reward.unsqueeze(1),(1,3,1)) # action, cursor, reward.
				new_boards[:,-4:, Axes.R_AX.value] = reward # action, cursor, reward.
			reward_preds = qfun_boards[:,-4:, 32+Axes.R_AX.value]
			pred_data = {'old_board':boards, 'new_board':new_boards, \
					'new_state_preds':qfun_boards,
					'rewards': reward[:,0], \
					'reward_preds': reward_preds[:,0],
					'w1':None, 'w2':None}
			loss = torch.sum((qfun_boards[:,:,33:64] - new_boards[:,:,1:32])**2) + \
				sum([torch.sum(1e-4 * torch.rand_like(param,dtype=g_dtype) * param * param) for param in qfun.parameters()])
			return loss
		
		loss = optimizer.step(closure)
		lloss = loss.detach().cpu().item()
		args["fd_losslog"].write(f'{uu}\t{lloss}\n')
		args["fd_losslog"].flush()
		if uu % 10 == 9:
			print(lloss)
		if uu % 25 == 0:
			updateMemory(memory_dict, pred_data)
			
		if uu % 1000 == 999: 
			qfun.save_checkpoint(f"checkpoints/{name}_{uu//1000}.pth")
		
	
def expandActionNodes(action_nodes, model, qfun, hcoo, reward_loc, locs, time):
	''' evaluate all the possible actions for a list of action_nodes
		by running the forward world model
		thereby predicting reward per action '''
	# first clean up the boards
	bs = len(action_nodes)
	ntok = token_cnt
	width = world_dim
	board = torch.zeros((bs, token_cnt, world_dim))
	for j in range(bs): 
		board[j,:,:32] = torch.tensor(action_nodes[j].board_enc)
	board = np.round(board * 4.0) / 4.0
	board = board.cuda()
		
	action_types,action_values = board_ops.enumerateActionList()
	nact = len(action_types)
	
	# check if we could guess here = cursor loc has no clue or guess
	# (the model predicts the can_guess flag @ 26+4)
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
		aenc = sparse_encoding.encodeActionNodes(at, av) # this is wasteful - only need one eval!
		s = aenc.shape[0] 
		# assert(s == 1)
		new_boards[:, i, 0:s, :] = aenc.cuda() # action is the first token

	new_boards = new_boards.reshape(bs * nact, ntok, width)
	boards_pred = model.forward(new_boards,hcoo,0,None)
	# mfun_pred = mfun.forward(boards_pred,None,0,None)
	# qfun_pred = qfun.forward(boards_pred,None,0,None)

	new_boards = new_boards.reshape(bs, nact, ntok, width)
	boards_pred = boards_pred.reshape(bs, nact, ntok, width)

	# mfun_pred = mfun_pred.detach().reshape(bs, nact, ntok, width)
	# mfun_pred = mfun_pred[:,:,reward_loc, 32+26].clone().squeeze() 
	# qfun_pred = qfun_pred.detach().reshape(bs, nact, ntok, width)
	# qfun_pred = qfun_pred[:,:,reward_loc, 32+26].clone().squeeze()
	reward_pred = boards_pred[:, :,reward_loc, 32+26].clone().squeeze()
	# copy over the beginning structure - needed!
	# this will have to be changed for longer-term memory TODO
	boards_pred[:,:,:,1:32] = boards_pred[:,:,:,33:]
	boards_pred[:,:,:,32:] = 0 # don't mess up forward computation
	boards_delta = boards_pred - new_boards
	# if can_guess[0] > 0.5:
	# 	plt.rcParams['toolbar'] = 'toolbar2'
	# 	fig,axs = plt.subplots(9,2,figsize=(30,20))
	# 	for i in range(9):
	# 		axs[i,0].imshow(boards_pred[0,i+4,:,:32].T.cpu().numpy())
	# 		axs[i,1].imshow(boards_delta[0,i+4,:,:32].T.cpu().numpy())
	# 	plt.show()
	# 	pdb.set_trace()

	boards_pred[:,:,reward_loc, Axes.R_AX.value] = 0 # it's a resnet - reset the reward.
	
	# save the one-step reward predictions. 
	for j in range(bs): 
		if len(reward_pred.shape) < 2: 
			pdb.set_trace()
		action_nodes[j].setRewardPred(reward_pred[j,:] )

	err = torch.sum((reward_pred > 3) * (reward_pred < 4.4))
	if err > 0:
		print(f'sanity check failed: {err} of the rewards are wrong. ')
		print(reward_pred * (reward_pred > 3) * (reward_pred < 4.5))
		print(torch.sum((reward_pred > 3) * (reward_pred < 4.5)))
		pdb.set_trace()
	# mask = reward_pred.clone() # torch.clip(reward_pred + 0.8, 0.0, 10.0)
	# 	# reward-weighted; if you can guess, generally do that.
	# valid_guesses = reward_pred[:,4:13] > 0
	
	action_node_new = []
	for j in range(bs):
		for i,(at,av) in enumerate(zip(action_types,action_values)):
			an = anode.ANode(at, av, reward_pred[j,i].item(), \
				boards_pred[j,i,:,:32], time)
			action_nodes[j].addKid(an)
		
	return action_nodes
	
def expandActionNodesAll(action_nodes, model, qfun, hcoo, reward_loc, locs, time):
	leaves = [] # should be a reference to 
	def recurse(an): 
		if len(an.kids) > 0: 
			for ann in an.kids: 
				recurse(ann)
		else: 
			# don't expand dead-ends or successes.
			if an.reward > -2.5 and an.reward < 2.5:
				leaves.append(an)
			
	for an in action_nodes: 
		recurse(an)
		
	# break up into batches. 
	n_leaves = len(leaves)
	print(f'expandActionNodesAll: working on a batch of {n_leaves}')
	with torch.no_grad():
		for i in range(0, len(leaves), 32):
			j = min(i + 32, len(leaves))
			expandActionNodes(leaves[i:j], model, qfun, hcoo, reward_loc, locs, time)
	return action_nodes
		
def expandActionNodesDepth(puzzles, model, qfun, hcoo, time, depth):
	bs = 8192
	pi = np.random.randint(0, puzzles.shape[0], (bs,))
	action_nodes = []
	puzzl_mat = np.zeros((bs,SuN,SuN))
	guess_mat = np.zeros((bs,SuN,SuN))
	print(f"expandActionNodesDepth: making a batch of {bs} rollouts")
	for k in range(bs):
		puzzl_mat[k,:,:] = puzzles[pi[k],:,:].numpy()
		curs_pos = torch.randint(SuN, (bs,2),dtype=int)
		board = torch.zeros(bs,token_cnt,world_dim)
		nodes,reward_loc,locs = sparse_encoding.sudokuToNodes(\
			puzzl_mat[k,:,:], guess_mat[k,:,:], curs_pos[k,:], 0, 0, 0.0) # action will be replaced.
		board[k,:,:],_,_ = sparse_encoding.encodeNodes(nodes)
		action_nodes.append( anode.ANode(8,0,0.0,board[k,:,:32],0) )

	for i in range(depth): 
		action_nodes = expandActionNodesAll(action_nodes, model, qfun, hcoo, reward_loc, locs, time)
	
	return action_nodes
	
	
def evaluateActionsBacktrack(model, mfun, qfun, puzzles, hcoo, nn):
	bs = 96
	pi = np.random.randint(0, puzzles.shape[0], (nn,bs))
	anode_list = []
	sudoku = Sudoku(SuN, SuK)
	for n in range(61,nn):
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
		rollouts_reward_pred = torch.zeros(duration, bs, 14)
		rollouts_action = torch.zeros(duration, bs, 2, dtype=int)
		rollouts_parent = torch.zeros(duration, bs, dtype=int)
		rollouts_board[0,:,:,:] = board[:,:,:32]
		board = board.numpy()
		sum_contradiction = torch.zeros(bs)
		# setup the root action nodes.
		root_nodes = [anode.ANode(8,0,0.0,board[k,:,:32],0) for k in range(bs)]
		action_nodes = [root_nodes[k] for k in range(bs)]
		# since the reward will be updated, keep a ref list of nodes
		rollout_nodes = [[None for k in range(bs)] for _ in range(duration)]
		for k in range(bs): 
			rollout_nodes[0][k] = root_nodes[k]
		
		for time in range(1,duration): 
			with torch.no_grad(): 
				board_new, action_node_new, reward_pred, contradiction, is_done = evaluateActions(model, mfun, qfun, board, hcoo, 0, reward_loc,locs, time, sum_contradiction, action_nodes)
				sum_contradiction = sum_contradiction + contradiction.cpu()
			board = board_new
			# root_nodes[0].print("")
			# backtracking!
			for k in range(bs):
				# default: replace with new node
				action_nodes[k] = action_node_new[k]
				if contradiction[k] > 0: 
					an = action_node_new[k] # there will never be alternatives here
					m = time # alternatives are added to the parents
					altern = an.getAltern()
					while m >= 0 and len(altern) < 1: 
						an.valid = False # eliminate it
						an_prev = an
						an = an.getParent()
						m = m-1
						altern = an.getAltern()
					if len(altern) == 0:
						pdb.set_trace() # puzzle must be solvable!
					# propagate the contradiction: this did not work! 
					an_prev.updateReward(-25)
					# could should sort by reward here. meh, take the first one.
					action_nodes[k] = altern[0]
					board[k,:,:32] = altern[0].board_enc
					board[k,:,32:] = 0.0 # jic
					# if k == 0:
					# 	print(colored(f"[{k}] backtracking to {action_nodes[k].index}", "blue"))
					# 	action_nodes[k].print("")
				# if k == 0: 
				# 	root_nodes[k].print("", all_actions=True)
				rollout_nodes[time][k] = action_nodes[k]

		for j in range(1,duration): 
			for k in range(bs):
				rollouts_board[j,k,:,:] = torch.tensor(rollout_nodes[j][k].board_enc, dtype=torch.float16)
				rollouts_reward[j,k] = rollout_nodes[j][k].reward
				rollouts_reward_pred[j,k,:] = rollout_nodes[j][k].reward_pred
				rollouts_action[j,k, 0] = rollout_nodes[j][k].action_type
				rollouts_action[j,k, 1] = rollout_nodes[j][k].action_value
				rollouts_parent[j,k] = rollout_nodes[j][k].getParent().index

		# in these files, the board is the state resulting from the action.
		# likewise for reward, which is updated through rollouts.
		# parent indexes the board prior the action.
		torch.save(rollouts_board, f'rollouts/rollouts_board_{n}.pt')
		torch.save(rollouts_reward, f'rollouts/rollouts_reward_{n}.pt')
		torch.save(rollouts_reward_pred, f'rollouts/rollouts_reward_pred_{n}.pt')
		torch.save(rollouts_action, f'rollouts/rollouts_action_{n}.pt')
		torch.save(rollouts_parent, f'rollouts/rollouts_parent_{n}.pt')

def moveValueDataset(puzzles, hcoo, bs, nn):
	''' for training the 'mouseizer':
		calculate the value of each square
		as distance to closest empty square
		then calculate the value of random moves from random positions
		as the discrete derivative of this '''
	try:
		boards = torch.load(f'rollouts/move_boards.pt',weights_only=True)
		actions = torch.load(f'rollouts/move_actions.pt',weights_only=True)
		rewards = torch.load(f'rollouts/move_rewards.pt',weights_only=True)
		nn = 0
	except Exception as error:
		print(colored(f"could not load precomputed data {error}", "red"))

	if nn > 0:
		boards = torch.zeros(nn,bs,token_cnt,32)
		actions = torch.zeros(nn,bs,2)
		rewards = torch.zeros(nn,bs)

		filts = []
		for r in range(3,26,2): # 3, 5, 7, 9, 11, 13, 15, 17
			filt = torch.zeros(1,1,r,r)
			c = r // 2
			for i in range(r):
				for j in range(r):
					# filters are a bunch of r-sized diamonds.  
					if abs(i-c) + abs(j-c) <= r//2:
						filt[0,0,i,j] = 1.0
			filts.append(filt)

		n = 0
		for i_p in range(nn // 16):
			# make the movements hard: only a few empties.
			num_empty = np.random.randint(1,8, (bs,))
			# these puzzles are degenerate, but that's ok
			puzzl_mat = np.random.randint(1,10,(bs,SuN,SuN))
			for k in range(bs):
				ne = num_empty[k]
				indx = np.random.randint(0,9, (ne,2))
				lin = np.arange(0,ne)
				puzzl_mat[k,indx[lin,0],indx[lin,1]] = 0
			guess_mat = np.zeros((bs,SuN,SuN)) 
			# due to the 1-hot represenation, guess / puzzle are equivalent. 
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
			for i_c in range(4):
				curs_pos = torch.randint(SuN, (bs,2),dtype=int)
				for i_m in range(4): 
					# select a move, calculate value. cursor pos is [x,y]
					# x is hence row or up/down
					move = np.ones(bs) * i_m # up right down left
					xnoty = move % 2 == 0
					direct = (move // 2) * 2 - 1
					direct = direct * (xnoty*2-1) # y axis is inverted
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
					n = n + 1
				
				plt.rcParams['toolbar'] = 'toolbar2'
				fig,axs = plt.subplots(3,2,figsize=(30,20))
				for k in range(4): 
					axs[k//2,k%2].imshow(boards[n-4+k,0,:,:].T.cpu().numpy())
					if k >= 2: 
						dif = boards[n-4+k,0,:,:] - boards[n-4+k-2,0,:,:]
						axs[k//2+1,k%2].imshow(dif.T.cpu().numpy())
				plt.show()

		torch.save(boards, 'rollouts/move_boards.pt')
		torch.save(actions, 'rollouts/move_actions.pt')
		torch.save(rewards, 'rollouts/move_rewards.pt')

	return boards,actions,rewards

def expandCoordinateVector(coo, a2a):
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
	# add top-level attention
	all2all = torch.Tensor(a2a);

	kids2parents = kids2parents.cuda()
	parents2kids = parents2kids.cuda()
	self2self = self2self.cuda()
	all2all = all2all.cuda()
	hcoo = [(kids2parents,dst_mxlen_k2p), (parents2kids,dst_mxlen_p2k), \
		(self2self, dst_mxlen_s2s), all2all]

	return hcoo

def getLayerCoordinateVectors():
	sudoku = Sudoku(SuN, SuK)
	_,_,coo,a2a,_,reward_loc = board_ops.encodeBoard(sudoku, np.zeros((9,9)), np.zeros((9,9)), np.zeros((2,), dtype=int), 0, 0)

	hcoo = expandCoordinateVector(coo, a2a)

	_,_,coo,a2a,_,reward_loc = board_ops.encodeBoard(sudoku, np.zeros((9,9)), np.zeros((9,9)), np.zeros((2,), dtype=int), 0, 0, many_reward=False) # FIXME

	hcoo_m = expandCoordinateVector(coo, a2a)
	hcoo_m.append(None) # for dense attention.

	return hcoo, hcoo_m, reward_loc


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train sudoku world model")
	parser.add_argument('-a', action='store_true', help='use AdamW as the optimizer (as opposed to PSGD)')
	parser.add_argument('-c', action='store_true', help='clear, start fresh: for training world model from random single actions')
	parser.add_argument('-e', action='store_true', help='evaluate with backtracking, save rollout data')
	parser.add_argument('-t', action='store_true', help='train world model')
	parser.add_argument('-q', type=int, default=0, help='train Q function from backtracking rollouts.  Argument is the number of files to use; check with `ls -lashtr rollouts/`')
	parser.add_argument('-m', action='store_true', help='train Q function for movements')
	parser.add_argument('--inverse_wm', action='store_true', help='train the acausal world model')
	cmd_args = parser.parse_args()
	
	try: 
		puzzles = torch.load(f'puzzles_{SuN}_500000.pt',weights_only=True)
	except Exception as error:
		print(colored(f"could not load puzzles {error}", "red"))
		print(colored("please download the puzzles from https://drive.google.com/file/d/1_q7fK3ei7xocf2rqFjSd17LIAA7a_gp4/view?usp=sharing", "blue"))
		print(colored("> gdown https://drive.google.com/file/d/1_q7fK3ei7xocf2rqFjSd17LIAA7a_gp4/view?usp=sharing --fuzzy", "blue"))
	
	NUM_ITERS = 200000
	device = torch.device('cuda:0') 
	# use export CUDA_VISIBLE_DEVICES=1
	# to switch to another GPU
	torch.set_float32_matmul_precision('high')
	torch.backends.cuda.matmul.allow_tf32 = True
	args = {"NUM_ITERS": NUM_ITERS, "device": device}
	
	# get our train and test dataloaders
	train_dataloader, test_dataloader = getDataLoaders(puzzles)

	hcoo,hcoo_m,reward_loc = getLayerCoordinateVectors()
	
	# allocate memory
	memory_dict = getMemoryDict()
	
	# define model 
	model = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=n_heads, n_layers=8, repeat=5, mode=0).to(device)
	model.printParamCount()
	
	# movement predictor
	# mfun = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=8, n_layers=8, repeat=2, mode=0).to(device)
	mfun = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=4, n_layers=10, repeat=3, mode=0).to(device)
	# works ok - does not converge but gets reasonable loss c.f. larger model.
	mfun.printParamCount()

	# qfun predictor
	qfun = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=6, n_layers=4, repeat=3, mode=0).to(device)
	# as of July 30 2024, this still does not train!
	qfun.printParamCount()

	if cmd_args.a:
		optimizer_name = "adamw"
	else:
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
					print(colored(f"could not load {prefix} model checkpoint", "red"))
					raise ValueError("No models found in the checkpoint directory")
				# Find the most recently modified file
				latest_file = max(files, key=os.path.getmtime)
				print(colored(latest_file, "green"))
				return latest_file

			if cmd_args.inverse_wm: 
				fname = "racoonizer_inv*"
			else: 
				fname = "racoonizer*"
			model.loadCheckpoint(getLatestFile(fname))
			print(colored("loaded model checkpoint", "blue"))

			mfun.loadCheckpoint(getLatestFile("mouse*"))
			print(colored("loaded mfun checkpoint", "blue"))

			qfun.loadCheckpoint(getLatestFile("quail*"))
			print(colored("loaded qfun checkpoint", "blue"))
			time.sleep(1)
		except Exception as error:
			print(error)

	if cmd_args.e: 
		anode_list = expandActionNodesDepth(puzzles, model, qfun, hcoo, 0, 5)
				
	if cmd_args.m:
		# train a (movement) policy
		anode_list = expandActionNodesDepth(puzzles, model, qfun, hcoo, 0, 2)
		
		# train a basic policy function: input is the state, output is 
		# advantage of that action over others. 
		for node in anode_list: 
			node.integrateReward()
			
		anode.outputGexf(anode_list[:12], f"anode.gexf")
			
		anode_flat = []
		for node in anode_list: 
			anode_flat = node.flattenNoLeaves(anode_flat)
		nn = len(anode_flat)
		rollouts_board = torch.zeros(nn,token_cnt,32)
		rollouts_reward = torch.zeros(nn,13)
		for i,node in enumerate(anode_flat): 
			rollouts_board[i,:,:] = torch.tensor(node.board_enc)
			rollouts_reward[i,:] = torch.tensor(node.horizon_reward)

		optimizer = getOptimizer(optimizer_name, mfun)

		# get the locations of the reward node.
		_,reward_loc,locs = sparse_encoding.sudokuToNodes(torch.zeros(9,9),torch.zeros(9,9),torch.zeros(2,dtype=int),0,0,0.0)

		fd_losslog = open('losslog.txt', 'w')
		args['fd_losslog'] = fd_losslog
		trainPolicy(rollouts_board, rollouts_reward, 300000, memory_dict, model, mfun, hcoo, hcoo_m, reward_loc, locs, "mouseizer")

	if cmd_args.q > 0:
		# Enables the matplotlib toolbar & thereby inspection
		plt.rcParams['toolbar'] = 'toolbar2'
		bs = 96
		nfiles = 62
		rollouts_board = torch.zeros(duration, bs*nfiles, token_cnt, 32, dtype=torch.float16)
		rollouts_parent_board = torch.zeros_like(rollouts_board)
		rollouts_reward = torch.zeros(duration, bs*nfiles)
		rollouts_action = torch.zeros(duration, bs*nfiles, 2, dtype=int)
		rollouts_parent = torch.zeros(duration, bs*nfiles, dtype=int)
		
		for i in range(nfiles): 
			r_board = torch.load(f'rollouts/rollouts_board_{i}.pt',weights_only=True)
			r_reward = torch.load(f'rollouts/rollouts_reward_{i}.pt',weights_only=True)
			r_rewardp = torch.load(f'rollouts/rollouts_reward_pred_{i}.pt',weights_only=True)
			r_action = torch.load(f'rollouts/rollouts_action_{i}.pt',weights_only=True)
			r_parent = torch.load(f'rollouts/rollouts_parent_{i}.pt',weights_only=True)
			
			rollouts_board[:,bs*i:bs*(i+1),:,:] = r_board
			rollouts_reward[:,bs*i:bs*(i+1)] = r_reward
			rollouts_action[:,bs*i:bs*(i+1),:] = r_action
			for j in range(duration):
				lin = torch.arange(bs)
				rollouts_parent_board[j,lin+bs*i,:,:] = \
					r_board[r_parent[j,lin],lin,:,:] 
			print(f"loaded rollouts/board - reward - action {i} .pt")
			
			# pdb.set_trace()
			# for j in range(bs//2): 
			# 	fig,axs = plt.subplots(3, 1, figsize=(32, 12))
			# 	im = axs[0].imshow(r_rewardp[:,j*2,:].squeeze().T.numpy())
			# 	plt.colorbar(im, ax=axs[0])
			# 	axs[0].set_title('reward_pred')
			# 	axs[1].plot(r_reward[:,j*2].squeeze().numpy())
			# 	axs[1].set_title('action_reward')
			# 	axs[2].plot(r_action[:,j*2,1].squeeze().numpy())
			# 	axs[2].set_title('guess')
			# 	plt.show()

		# flatten to get uniform actions
		rollouts_board = rollouts_board.reshape(duration*bs*nfiles, token_cnt, 32)
		rollouts_parent_board = rollouts_parent_board.reshape(duration*bs*nfiles, token_cnt, 32)
		rollouts_reward = rollouts_reward.reshape(duration*bs*nfiles)
		rollouts_action = rollouts_action.reshape(duration*bs*nfiles, 2)

		# need to select only guess actions --
		# the moves are handled by mouseizer.
		# pdb.set_trace()
		guess_index = (rollouts_action[:,0] == 4).nonzero().squeeze()
		rollouts_board = rollouts_board[guess_index, :, :]
		rollouts_parent_board = rollouts_parent_board[guess_index, :, :]
		rollouts_reward = rollouts_reward[guess_index]
		rollouts_reward = torch.clip(rollouts_reward, -5, 5)
		rollouts_action = rollouts_action[guess_index,:]
		
		# for i in range(4):
		# 	j = np.random.randint(guess_index.shape[0])
		# 	fig,axs = plt.subplots(2,1, figsize=(20,16))
		# 	axs[0].imshow(rollouts_parent_board[j,:,:].T.numpy())
		# 	axs[1].imshow(rollouts_board[j,:,:].T.numpy())
		# 	print(rollouts_action[j,:])
		# 	plt.show()
		
		# copy the action to the parent board for the model forward pass
		# (the cursor position should already be updated)
		rollouts_parent_board[:,0,:] = rollouts_board[:,0,:]

		optimizer = getOptimizer(optimizer_name, qfun)

		# get the locations of the board nodes.
		_,reward_loc,locs = sparse_encoding.sudokuToNodes(torch.zeros(9,9),torch.zeros(9,9),torch.zeros(2,dtype=int),0,0,0.0)

		fd_losslog = open('losslog.txt', 'w')
		args['fd_losslog'] = fd_losslog
		trainQfun(rollouts_parent_board, rollouts_reward, rollouts_action, 1300000, memory_dict, model, qfun, hcoo, reward_loc, locs, "quailizer")

	if cmd_args.t:
		uu = 0
		fd_losslog = open('losslog.txt', 'w')
		args['fd_losslog'] = fd_losslog
		while uu < NUM_ITERS:
			uu = train(args, memory_dict, model, train_dataloader, optimizer, hcoo, reward_loc, uu, cmd_args.inverse_wm)
 
		# print("validation")
		validate(args, model, test_dataloader, optimizer_name, hcoo, uu, cmd_args.inverse_wm)
