import math
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pdb
import matplotlib.pyplot as plt
import sparse_encoding
from gracoonizer import Gracoonizer
from sudoku_gen import Sudoku
from plot_mmap import make_mmf, write_mmap
from netdenoise import NetDenoise
from test_gtrans import getTestDataLoaders, SimpleMLP
from constants import *
from type_file import Action
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
			reward = -0.25
			
	if True: 
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
	board encoding: Shape (#board nodes x 20)
	action encoding: Shape (#action nodes x 20)
	new board encoding: Shape (#newboard nodes x 20)
	msk: Shape (#board&action nodes x #board&action) represents nodes parent/child relationships
		which defines the attention mask used in the transformer heads
	'''
	nodes, reward_loc = sparse_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, action, action_val, 0.0)
	benc,coo = sparse_encoding.encodeNodes(nodes)
	
	reward = runAction(sudoku, guess_mat, curs_pos, action, action_val)
	
	nodes, reward_loc = sparse_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, action, -1, reward) # action_val doesn't matter
	newbenc,coo = sparse_encoding.encodeNodes(nodes)
	
	return benc, newbenc, coo, reward, reward_loc


def generateActionValue(action: int, min_dist: int, max_dist: int):
	'''
	Generates an action value corresponding to the action.
	For movement actions, samples a dist unif on [min_dist, max_dist] and 
		chooses - or + direction based on the action (ex: -1 for left, +1 for right).

	min_dist: (int) Represents the min distance travelled.
	max_dist: (int) Represents the max distance travelled.
	'''
	# movement action
	dist = np.random.randint(low=min_dist, high=max_dist+1)
	if action in [Action.DOWN.value,Action.LEFT.value]:
		direction = -1
		return dist * direction 

	if action in [Action.UP.value, Action.RIGHT.value]:
		direction = 1
		return dist * direction 

	# guess or set note action
	if action in [Action.SET_GUESS.value, Action.SET_NOTE.value]:
		return np.random.randint(1,10)

	# nop
	return 0

	
def enumerateMoves(depth, episode, possible_actions=[]): 
	if not possible_actions:
		possible_actions = [ 0,1,2,3,4,5 ]
		possible_actions.append(Action.SET_GUESS.value) # upweight
		possible_actions.append(Action.SET_GUESS.value)
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
	lst = enumerateMoves(1, [], possible_actions)
	if len(lst) < n: 
		rep = n // len(lst) + 1
		lst = lst * rep
	if len(lst) > n: 
		lst = random.sample(lst, n)
	sudoku = Sudoku(SuN, SuK)
	orig_boards = [] 
	new_boards = []
	actions = []
	masks = []
	rewards = torch.zeros(n)
	guess_mat = np.zeros((SuN, SuN))
	for i, ep in enumerate(lst): 
		puzzl = puzzles[i, :, :]
		sudoku.setMat(puzzl.numpy())
		curs_pos = torch.randint(SuN, (2,))
		action_val = generateActionValue(ep[0], min_dist, max_dist)
		
		# benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, ep[0], action_val)
		benc,newbenc,coo,reward,reward_loc = encodeBoard(sudoku, guess_mat, curs_pos, ep[0], action_val)
		orig_boards.append(torch.tensor(benc))
		new_boards.append(torch.tensor(newbenc))
		rewards[i] = reward
		
	orig_board_enc = torch.stack(orig_boards)
	new_board_enc = torch.stack(new_boards)
	return orig_board_enc, new_board_enc, coo, rewards, reward_loc

def trainValSplit(data_matrix: torch.Tensor, num_eval=None, eval_ratio: float = 0.2):
	'''
	Split data matrix into train and val data matrices
	data_matrix: (torch.tensor) Containing rows of data
	num_eval: (int) If provided, is the number of rows in the val matrix
	'''
	num_samples = data_matrix.size(0)
	if num_samples <= 1:
		raise ValueError(f"data_matrix needs to be a tensor with more than 1 row")

	if not num_eval:
		num_eval = int(num_samples * eval_ratio)
	
	training_data = data_matrix[:-num_eval]
	eval_data = data_matrix[-num_eval:]
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
			

def getDataLoaders(puzzles, num_samples, num_eval=2000):
	'''
	Returns a pytorch train and test dataloader
	'''
	data_dict, coo, reward_loc = getDataDict(puzzles, num_samples, num_eval)
	train_dataset = SudokuDataset(data_dict['train_orig_board'],
											data_dict['train_new_board'], 
											data_dict['train_rewards'])

	test_dataset = SudokuDataset(data_dict['test_orig_board'],
										data_dict['test_new_board'], 
										data_dict['test_rewards'])

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	return train_dataloader, test_dataloader, coo, reward_loc


def getDataDict(puzzles, num_samples, num_eval=2000):
	'''
	Returns a dictionary containing training and test data
	'''
	orig_board, new_board, coo, rewards, reward_loc = enumerateBoards(puzzles, num_samples)
	print(orig_board.shape, new_board.shape, rewards.shape)
	train_orig_board, test_orig_board = trainValSplit(orig_board, num_eval=num_eval)
	train_new_board, test_new_board = trainValSplit(new_board, num_eval=num_eval)

	train_rewards, test_rewards = trainValSplit(rewards, num_eval=num_eval)

	dataDict = {
		'train_orig_board': train_orig_board,
		'train_new_board': train_new_board,
		'train_rewards': train_rewards,
		'test_orig_board': test_orig_board,
		'test_new_board': test_new_board,
		'test_rewards': test_rewards
    }
	return dataDict, coo, reward_loc

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
	loss_mask = torch.ones(1, board_enc.shape[1], board_enc.shape[2], device=device)
	for i in range(11,20):
		loss_mask[:,:,i] *= 0.001 # semi-ignore the "latents"
	return loss_mask 

def getOptimizer(optimizer_name, model, lr=1e-3, weight_decay=0):
	if optimizer_name == "adam": 
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == 'adamw':
		optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	else: 
		optimizer = psgd.LRA(model.parameters(),lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=10, grad_clip_max_norm=5)
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
	write_mmap(memory_dict['fd_board'], pred_dict['old_board'][0:4,:,:].cpu())
	write_mmap(memory_dict['fd_new_board'], pred_dict['new_board'][0:4,:,:].cpu())
	write_mmap(memory_dict['fd_boardp'], pred_dict['new_state_preds'][0:4,:,:].cpu().detach())
	write_mmap(memory_dict['fd_reward'], pred_dict['rewards'][0:4].cpu())
	write_mmap(memory_dict['fd_rewardp'], pred_dict['reward_preds'][0:4].cpu().detach())
	write_mmap(memory_dict['fd_attention'], torch.stack((pred_dict['a1'], pred_dict['a2']), 0))
	write_mmap(memory_dict['fd_wqkv'], torch.stack((pred_dict['w1'], pred_dict['w2']), 0))
	return 

def train(args, memory_dict, model, train_loader, optimizer, hcoo, dst_mxlen, reward_loc, uu):
	model.train()
	sum_batch_loss = 0.0

	for batch_idx, batch_data in enumerate(train_loader):
		old_board, new_board, rewards = [t.to(args["device"]) for t in batch_data.values()]

		pred_data = {}
		if optimizer_name != 'psgd': 
			optimizer.zero_grad()
			new_state_preds,a1,a2,w1,w2 = \
				model.forward(old_board, hcoo, dst_mxlen, uu, None)
			reward_preds = new_state_preds[:,reward_loc, 20]
			pred_data = {'old_board':old_board, 'new_board':new_board, 'new_state_preds':new_state_preds,
					  		'rewards': rewards, 'reward_preds': reward_preds,
							'a1':a1, 'a2':a2, 'w1':w1, 'w2':w2}
			loss = torch.sum((new_state_preds[:,:,0:21] - new_board[:,:,0:21])**2) + \
					torch.sum((new_state_preds[:,:,21:] - new_board[:,:,21:])**2)*1e-4 + \
					sum( \
					[torch.sum(1e-4 * torch.rand_like(param) * param * param) for param in model.parameters()])
			loss.backward()
			optimizer.step() 
			print(loss.detach().cpu().item())
		else: 
			# psgd library internally does loss.backwards and zero grad
			def closure():
				nonlocal pred_data
				new_state_preds, a1,a2,w1,w2 = model.forward(old_board, hcoo, dst_mxlen, uu, None)
				reward_preds = new_state_preds[:,reward_loc, 20]
				pred_data = {'old_board':old_board, 'new_board':new_board, 'new_state_preds':new_state_preds,
					  		'rewards': rewards, 'reward_preds': reward_preds,
							'a1':a1, 'a2':a2, 'w1':w1, 'w2':w2}
				loss = torch.sum((new_state_preds[:,:,0:21] - new_board[:,:,0:21])**2) + \
					torch.sum((new_state_preds[:,:,21:] - new_board[:,:,21:])**2)*1e-4 + \
					sum( \
					[torch.sum(1e-4 * torch.rand_like(param) * param * param) for param in model.parameters()])
					# we seem to have lost the comment explaining why this was here 
					# but it was recommended by the psgd authors to break symmetries w a L2 norm on the weights. 
				return loss
			loss = optimizer.step(closure)
		
		lloss = loss.detach().cpu().item()
		print(lloss)
		args["fd_losslog"].write(f'{uu}\t{lloss}\n')
		args["fd_losslog"].flush()
		uu = uu + 1

		sum_batch_loss += lloss
		if batch_idx % 25 == 0:
			updateMemory(memory_dict, pred_data)
			pass 
	
	# add epoch loss
	avg_batch_loss = sum_batch_loss / len(train_loader)
	return uu
	
	
def validate(args, model, test_loader, optimzer_name, hcoo, dst_mxlen, uu):
	model.eval()
	sum_batch_loss = 0.0
	with torch.no_grad():
		for batch_data in test_loader:
			old_board, new_board, rewards = [t.to(args["device"]) for t in batch_data.values()]
			new_state_preds,a1,a2,w1,w2 = model.forward(old_board, hcoo, dst_mxlen, uu, None)
			reward_preds = new_state_preds[:,reward_loc, 20]
			loss = torch.sum((new_state_preds[:,:,0:21] - new_board[:,:,0:21])**2)
			lloss = loss.detach().cpu().item()
			print(f'v{lloss}')
			fd_losslog.write(f'{uu}\t{lloss}\n')
			fd_losslog.flush()
			sum_batch_loss += loss.cpu().item()
	
	avg_batch_loss = sum_batch_loss / len(test_loader)
			
	
	return 

if __name__ == '__main__':
	puzzles = torch.load('puzzles_500000.pt')
	NUM_SAMPLES = batch_size * 200 # must be a multiple, o/w get bumps in the loss from the edge effects of dataloader enumeration
	NUM_EVAL = batch_size * 25
	NUM_EPOCHS = 500
	device = torch.device('cuda:0')
	fd_losslog = open('losslog.txt', 'w')
	args = {"NUM_SAMPLES": NUM_SAMPLES, "NUM_EPOCHS": NUM_EPOCHS, "NUM_EVAL": NUM_EVAL, "device": device, "fd_losslog": fd_losslog}
	
	optimizer_name = "psgd" # adam, adamw, or psgd
	
	# get our train and test dataloaders
	train_dataloader, test_dataloader, coo, reward_loc = getDataLoaders(puzzles, args["NUM_SAMPLES"], args["NUM_EVAL"])
	
	print(coo)
	# # # full coo
	# i = 0
	# coo = torch.zeros(10, 2, dtype=int)
	# for dst in range(5): 
	# 	for src in range(5): 
	# 		if src > dst: 
	# 			coo[i, 0] = dst
	# 			coo[i, 1] = src
	# 			i = i + 1
	# print(coo)
	# # we know that the upper triangle works, but super sparse does not. 
	# # gradually remove links until it stops working. 
	# co = []
	# co.append([0,1]) #orig not needed & distractor
	# co.append([2,0]) #orig not needed
	# # co.append([0,3]) # not needed
	# # co.append([0,4]) # nn
	# # co.append([1,2]) # nn
	# # co.append([1,3]) # absolutely essential! works slower with 0,3
	# # co.append([1,4]) # absolutely essential! works slower with 0,4
	# co.append([3,2]) #orig (functions without one of 2,3 or 2,4, but poorly)
	# co.append([4,2]) #orig
	# # co.append([3,4]) # nn
	# co.append([2,2])
	# co.append([3,0])
	# co.append([4,0])
	# coo = torch.tensor(co)
	
	# first half of heads are kids to parents
	kids2parents, dst_mxlen_f, src_mxlen_f = expandCoo(coo)
	dst_mxlen = dst_mxlen_f
	# swap dst and src
	coo_ = torch.zeros_like(coo)
	coo_[:,0] = coo[:,1]
	coo_[:,1] = coo[:,0]
	parents2kids, dst_mxlen_b, src_mxlen_b = expandCoo(coo_)
	dst_mxlen = max(dst_mxlen_b, dst_mxlen)
	# per-head (sorta) COO vector
	hcoo = torch.stack([kids2parents, parents2kids]).to('cuda')
	
	# allocate memory
	memory_dict = getMemoryDict()
	
	# define model 
	model = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, reward_dim=1).to(device)
	model.printParamCount()
	try: 
		#model.load_checkpoint()
		#print("loaded model checkpoint")
		pass 
	except : 
		print("could not load model checkpoint")
	
	optimizer = getOptimizer(optimizer_name, model)

	uu = 0
	for _ in range(0, args["NUM_EPOCHS"]):
		uu = train(args, memory_dict, model, train_dataloader, optimizer, hcoo, dst_mxlen, reward_loc, uu)
	
	# save after training
	model.save_checkpoint()

	print("validation")
	validate(args, model, test_dataloader, optimizer_name, hcoo, dst_mxlen, reward_loc, uu)
