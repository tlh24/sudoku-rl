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
import anode
import board_ops
import psgd 
	# https://sites.google.com/site/lixilinx/home/psgd
	# https://github.com/lixilinx/psgd_torch/issues/2

from utils import set_seed
# from diffusion.data import getBaselineDataloaders
# from diffusion.model import GPTConfig, GPT

sys.path.insert(0, "baseline/")


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
	orig_board, new_board, coo, a2a, rewards, reward_loc = board_ops.enumerateBoards(puzzles, num_samples)
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
	''' this trains the *world model* on random actions placed
	on random boards '''

	for batch_idx, batch_data in enumerate(train_loader):
		old_board, new_board, rewards = [t.to(args["device"]) for t in batch_data.values()]
		
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

		if batch_idx % 25 == 0:
			updateMemory(memory_dict, pred_data)
			pass 
	
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

def main_(args):
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
	
def trainQfun(rollouts_board, rollouts_reward, rollouts_action, nn, memory_dict, model, qfun, hcoo, reward_loc, locs, name):
	n_roll = rollouts_board.shape[0]
	n_tok = rollouts_board.shape[1]
	width = rollouts_board.shape[2]
	pred_data = {}
	for uu in range(nn): 
		indx = torch.randint(0,n_roll,(batch_size,))
		boards = rollouts_board[indx,:,:].squeeze().float()
		reward = rollouts_reward[indx].squeeze()
		actions = rollouts_action[indx,:].squeeze() # not used! 
		# already encoded in the board! 
		
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
			# plt.plot((model_boards[:,reward_loc, 32+26] - reward).cpu().numpy())
			# pdb.set_trace()
		def closure(): 
			nonlocal pred_data
			if name == 'mouseizer' or name == 'quailizer':
				hcoo2 = None # None works ok with movements.
				# initializing from the bare board does not work well..
			else:
				hcoo2 = hcoo
			qfun_boards,_,_ = qfun.forward(model_boards,hcoo2,0,None)
			reward_preds = qfun_boards[:,reward_loc, 32+26]
			pred_data = {'old_board':boards, 'new_board':model_boards, 'new_state_preds':qfun_boards,
								'rewards': reward, 'reward_preds': reward_preds,
								'w1':None, 'w2':None}
			loss = torch.sum((reward - reward_preds)**2) + \
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
		
	
def evaluateActions(model, mfun, qfun, board, hcoo, depth, reward_loc, locs, time, sum_contradiction, action_nodes):
	''' evaluate all the possible actions for a current board
		by running the forward world model
		thereby predicting reward per action '''
	# first clean up the boards
	board = np.round(board * 4.0) / 4.0
	board = torch.tensor(board).cuda()
	bs = board.shape[0]
	ntok = board.shape[1]
	width = board.shape[2]
		
	action_types,action_values = board_ops.enumerateActionList(9+4+1)
		# inculdes 'unguess' action - presently unnecessary
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
		s = aenc.shape[0] # should be 1
		new_boards[:, i, 0:s, :] = aenc.cuda()

	new_boards = new_boards.reshape(bs * nact, ntok, width)
	boards_pred,_,_ = model.forward(new_boards,hcoo,0,None)
	mfun_pred,_,_ = mfun.forward(boards_pred,None,0,None)
	# qfun_pred,_,_ = qfun.forward(boards_pred,None,0,None)

	boards_pred = boards_pred.detach().reshape(bs, nact, ntok, width)
	mfun_pred = mfun_pred.detach().reshape(bs, nact, ntok, width)
	mfun_pred = mfun_pred[:,:,reward_loc, 32+26].clone().squeeze()
	# qfun_pred = qfun_pred.detach().reshape(bs, nact, ntok, width)
	# qfun_pred = qfun_pred[:,:,reward_loc, 32+26].clone().squeeze()
	reward_pred = boards_pred[:, :,reward_loc, 32+26].clone().squeeze()
	# copy over the beginning structure - needed!
	# this will have to be changed for longer-term memory TODO
	boards_pred[:,:,:,1:32] = boards_pred[:,:,:,33:]
	boards_pred[:,:,:,32:] = 0 # don't mess up forward computation
	boards_pred[:,:,reward_loc, 26] = 0 # it's a resnet - reset the reward.
	
	# save the one-step reward predictions. 
	for j in range(bs): 
		action_nodes[j].setRewardPred(reward_pred[j,:])

	mask = reward_pred.clone() # torch.clip(reward_pred + 0.8, 0.0, 10.0)
		# reward-weighted; if you can guess, generally do that.
	valid_guesses = reward_pred[:,4:13] > 0
	mask[:,0:4] = mfun_pred[:,0:4] * 2
	# mask[:,4:] = mask[:,4:] + qfun_pred[:,4:] # add before sm -> multiply
	mask = F.softmax(mask, 1)
	# mask = mask / torch.sum(mask, 1).unsqueeze(1).expand(-1,14)
	indx = torch.multinomial(mask, 1).squeeze()

	lin = torch.arange(0,bs)
	boards_pred_taken = boards_pred[lin,indx,:,:].detach().squeeze().cpu().numpy()
	contradiction = can_guess * \
		(1-torch.clip(torch.sum(reward_pred[:,4:13] > 0, 1), 0, 1))
	# detect a contradiction when the square is empty = can_guess is True,
	# but no actions expected to result in reward.
	# would be nice if this was learned, not hand-coded..
	
	reward_pred_taken = reward_pred[lin,indx]
	reward_np = reward_pred_taken
	
	action_node_new = []
	indent = " " # * depth
	for j in range(bs):
		at_t = action_types[indx[j]]
		av_t = action_values[indx[j]]
		an_taken = anode.ANode(at_t, av_t, reward_pred_taken[j], boards_pred_taken[j,:,:32].squeeze(), time)
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
					an = anode.ANode(at, av, reward_pred[j,m+4], boards_pred[j,m+4,:,:32].cpu().numpy().squeeze(), time)
					action_nodes[j].addKid(an)
		if j == 0: 
			print(f"{time} action {getActionName(at_t)} {av_t} reward {reward_pred_taken[0].item()}")
			if contradiction[0] > 0.5:
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
		
	return boards_pred_taken, action_node_new, reward_pred, contradiction, is_done

	
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
	parser.add_argument('-a', action='store_true', help='use AdamW as the optimizer (as opposed to PSGD)')
	parser.add_argument('-c', action='store_true', help='clear, start fresh: for training world model from random single actions')
	parser.add_argument('-e', action='store_true', help='evaluate with backtracking, save rollout data')
	parser.add_argument('-t', action='store_true', help='train world model')
	parser.add_argument('-q', type=int, default=0, help='train Q function from backtracking rollouts.  Argument is the number of files to use; check with `ls -lashtr rollouts/`')
	parser.add_argument('-m', action='store_true', help='train Q function for movements')
	cmd_args = parser.parse_args()
	
	puzzles = torch.load(f'puzzles_{SuN}_500000.pt')
	NUM_TRAIN = 64 * 1800
	NUM_VALIDATE = 64 * 300
	NUM_SAMPLES = NUM_TRAIN + NUM_VALIDATE
	NUM_ITERS = 100000
	device = torch.device('cuda:0') 
	# use export CUDA_VISIBLE_DEVICES=1
	# to switch to another GPU
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
	# mfun = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=8, n_layers=8, repeat=2, mode=0).to(device)
	mfun = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=4, n_layers=4, repeat=1, mode=0).to(device)
	# works ok - does not converge but gets reasonable loss c.f. larger model.
	# this is an all-to-all transformer; see line 492
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
		anode_list = evaluateActionsBacktrack(model, mfun, qfun, puzzles, hcoo, 319)
				
	if cmd_args.m:
		bs = 96
		nn = 1500
		rollouts_board,rollouts_action,rollouts_reward = moveValueDataset(puzzles, hcoo, bs,nn)

		optimizer = getOptimizer(optimizer_name, mfun)

		# get the locations of the reward node.
		_,reward_loc,locs = sparse_encoding.sudokuToNodes(torch.zeros(9,9),torch.zeros(9,9),torch.zeros(2,dtype=int),0,0,0.0)

		rollouts_board = rollouts_board.reshape(nn*bs,token_cnt,32)
		rollouts_action = rollouts_action.reshape(nn*bs,2)
		rollouts_reward = rollouts_reward.reshape(nn*bs)

		fd_losslog = open('losslog.txt', 'w')
		args['fd_losslog'] = fd_losslog
		trainQfun(rollouts_board, rollouts_reward, rollouts_action, 300000, memory_dict, model, mfun, hcoo, reward_loc, locs, "mouseizer")
		# note: no hcoo; only all-to-all attention

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
			r_board = torch.load(f'rollouts/rollouts_board_{i}.pt')
			r_reward = torch.load(f'rollouts/rollouts_reward_{i}.pt')
			r_rewardp = torch.load(f'rollouts/rollouts_reward_pred_{i}.pt')
			r_action = torch.load(f'rollouts/rollouts_action_{i}.pt')
			r_parent = torch.load(f'rollouts/rollouts_parent_{i}.pt')
			
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
