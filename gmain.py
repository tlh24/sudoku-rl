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
from tqdm import tqdm
import time 
import sys 
import argparse 
from type_file import Action, Axes, getActionName
from l1attn_sparse_cuda import expandCoo
import psgd 
	# https://sites.google.com/site/lixilinx/home/psgd
	# https://github.com/lixilinx/psgd_torch/issues/2

from utils import set_seed
# from diffusion.data import getBaselineDataloaders
# from diffusion.model import GPTConfig, GPT

sys.path.insert(0, "baseline/")

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

def oneHotEncodeBoard(sudoku, curs_pos, action: int, action_val: int, enc_dim: int = 20):
	'''
	Note: Assume that action is a movement action and that we have 2 dimensional sudoku 
	
	Encode the current pos as a euclidean vector [x,y],
		encode the action (movement) displacement as a euclidean vector [dx,dy],
		runs the action, encodes the new board state.
	Mask is hardcoded to match the graph mask generated from one bnode and one actnode
	'''
	# ensure two-dim sudoku
	if curs_pos.size(0) != 2:
		raise ValueError(f"Must have 2d sudoku board")

	# ensure that action is movement action
	if action not in [Action.DOWN.value, Action.UP.value, Action.LEFT.value, Action.RIGHT.value]:
		raise ValueError(f"The action must be a movement action but received: {action}")

	if action in [Action.DOWN.value, Action.UP.value]:
		action_enc = np.array([0, action_val], dtype=np.float32).reshape(1,-1)
	else:
		action_enc = np.array([action_val, 0], dtype=np.float32).reshape(1,-1)
	
	curs_enc = curs_pos.numpy().astype(np.float32).reshape(1,-1)

	# right pad with zeros to encoding dimension
	action_enc = np.pad(action_enc, ((0,0), (0, enc_dim-action_enc.shape[1])))
	curs_enc = np.pad(curs_enc, ((0,0), (0, enc_dim - curs_enc.shape[1])))
	assert(enc_dim == action_enc.shape[1] == curs_enc.shape[1])

	# hard code mask to match the mask created by one board node, one action node
	mask = np.full((2,2), 8.0, dtype=np.float32)
	np.fill_diagonal(mask, 1.0)
	
	reward = runAction(sudoku, None, curs_pos, action, action_val)
	
	new_curs_enc = curs_enc + action_enc  
	
	return curs_enc, action_enc, new_curs_enc, mask, reward

	
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
	Returns a pytorch train and test dataloader for gracoonizer position prediction
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
			if is_train and batch_idx % 25 == 0:
				updateMemory(self.memory_dict, pred_data)
				pass 
		
		# add epoch loss
		avg_batch_loss = sum_batch_loss / len(data_loader)
		self.fd_loss_log.write(f'{epoch}\t{avg_batch_loss}\n')
		self.fd_loss_log.flush()
		return 

	def train(self):
		self.model.train()
		for epoch_num in tqdm(range(0, self.args.epochs)):
			self.forwardLoop(epoch_num, True, self.train_dl)

	def validate(self):
		self.model.eval()
		with torch.no_grad():
			self.forwardLoop(self.args.epochs, False, self.test_dl)
		
# class RecurrentBaselineTrainer(Trainer):
# 	'''
# 	Used for the recurrent transformer baseline. 
# 	'''
# 	def __init__(self, model, train_dl, test_dl, device, optimizer_name, args, loss_log_path='losslog.txt'):
# 		super().__init__(model, train_dl, test_dl, device, optimizer_name, None, args, None, loss_log_path)
# 	
# 	def forwardLoop(self, epoch, is_train, data_loader):
# 		sum_batch_loss = 0.0
# 
# 		for batch_idx, (x,y) in enumerate(data_loader):
# 			x = x.to(self.device)
# 			y = y.to(self.device)
# 
# 			# forward the model
# 			with torch.set_grad_enabled(is_train):
# 				logits, loss, atts = self.model(x,y)
# 				loss = loss.mean()
# 				sum_batch_loss += loss.item()
# 
# 			if is_train:
# 				self.model.zero_grad()
# 				loss.backward()
# 				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
# 				self.optimizer.step()
# 		
# 			sum_batch_loss += loss.cpu().item()
# 		
# 		# add epoch loss to file
# 		avg_batch_loss = sum_batch_loss / len(data_loader)
# 		self.fd_loss_log.write(f'{epoch}\t{avg_batch_loss}\n')
# 		self.fd_loss_log.flush()
# 		return 
			
			
## graph-baseline

def main(args):
	start_time = time.time() 

	# seed for reproducibility
	set_seed(42)	

	device = torch.device('cuda:0')
	optimizer_name = "adam" # or psgd
	torch.set_float32_matmul_precision('high')

	if args.gracoonizer:
		puzzles = torch.load('puzzles_500000.pt')
		
		# get our train and test dataloaders
		train_dataloader, test_dataloader = getDataLoaders(puzzles, args.n_train + args.n_test, args.n_test)
		
		# allocate memory
		memory_dict = getMemoryDict()
		
		# define model 
		model = Gracoonizer(xfrmr_dim = 20, world_dim = 20, reward_dim = 1)
		model.printParamCount()
		try: 
			#model.load_checkpoint()
			#print("loaded model checkpoint")
			pass 
		except : 
			print("could not load model checkpoint")

		criterion = nn.MSELoss()

		trainer = Trainer(model, train_dataloader, test_dataloader, device, optimizer_name, criterion,args, memory_dict)
	
	else:
		# get our train and test dataloaders
		train_dataloader, test_dataloader = getBaselineDataloaders(args)

		# define model 
		mconf = GPTConfig(vocab_size=10, block_size=81, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
        num_classes=9, causal_mask=False, losses=args.loss, n_recur=args.n_recur, all_layers=args.all_layers,
        hyper=args.hyper)
		
		model = GPT(mconf)

		trainer = RecurrentBaselineTrainer(model, train_dataloader, test_dataloader, device, optimizer_name,args)

	# training 
	trainer.train()

	# save after training
	#trainer.saveCheckpoint()

	# validation
	print("validation")
	trainer.validate()
	end_time = time.time()
	program_duration = end_time - start_time
	print(f"Program duration: {program_duration} sec")

	
## graph-baseline 

# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	# Training
# 	parser.add_argument('--epochs', type=int, default=200)
# 	parser.add_argument('--eval_interval', type=int, default=1, help='Compute eval for how many epochs')
# 	parser.add_argument('--batch_size', type=int, default=128)
# 	parser.add_argument('--lr', type=float, default=6e-4)
# 	parser.add_argument('--lr_decay', default=False, action='store_true')
# 
# 	# Model and loss
# 	parser.add_argument('--n_layer', type=int, default=1, help='Number of sequential self-attention blocks.')
# 	parser.add_argument('--n_recur', type=int, default=32, help='Number of recurrency of all self-attention blocks.')
# 	parser.add_argument('--n_head', type=int, default=4, help='Number of heads in each self-attention block.')
# 	parser.add_argument('--n_embd', type=int, default=128, help='Vector embedding size.')
# 	parser.add_argument('--loss', default=[], nargs='+', help='specify regularizers in \{c1, att_c1\}')
# 	parser.add_argument('--all_layers', default=False, action='store_true', help='apply losses to all self-attention layers')    
# 	parser.add_argument('--hyper', default=[1, 0.1], nargs='+', type=float, help='Hyper parameters: Weights of [L_sudoku, L_attention]')
# 
# 	# Data
# 	parser.add_argument('--n_train', type=int, default=10000, help='The number of data for training')
# 	parser.add_argument('--n_test', type=int, default=2000, help='The number of data for testing')
# 
# 	# Other
# 	parser.add_argument('--seed', type=int, default=0, help='Random seed for reproductivity.')
# 	parser.add_argument('--gpu', type=int, default=-1, help='gpu index; -1 means using all GPUs or using CPU if no GPU is available')
# 	parser.add_argument("--gracoonizer", action=argparse.BooleanOptionalAction, default=True)
# 	args = parser.parse_args()
# 
# 	main(args)

## main

