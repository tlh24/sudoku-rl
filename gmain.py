import math
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pdb
import matplotlib.pyplot as plt
import graph_encoding
from gracoonizer import Gracoonizer
from sudoku_gen import Sudoku
from plot_mmap import make_mmf, write_mmap
from netdenoise import NetDenoise
from constants import *
from type_file import Action
import psgd 
	# https://sites.google.com/site/lixilinx/home/psgd
	# https://github.com/lixilinx/psgd_torch/issues/2

def runAction(action: int, sudoku, curs_pos): 
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
	curs_pos[0] = curs_pos[0] % SuN # wrap at the edges; 
	curs_pos[1] = curs_pos[1] % SuN # works for negative nums
			
	if True: 
		print(f'runAction @ {curs_pos[0]},{curs_pos[1]}: {action}')
	
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
	
	reward = runAction(action, sudoku, curs_pos)
	
	new_curs_enc = curs_enc + action_enc  
	
	return curs_enc, action_enc, new_curs_enc, mask, reward


def encodeBoard(sudoku, curs_pos, action, action_val):  
	'''
	Encodes the current board state and encodes the given action,
		runs the action, and then encodes the new board state.
		Also returns a mask matrix (#nodes by #nodes) which represents parent/child relationships
		which defines the attention mask used in the transformer heads
	# TODO: support multiple actions

	The board and action nodes have the same encoding- contains one hot of node type and node value
	
	Returns:
	board encoding: Shape (#board nodes x 20)
	action encoding: Shape (#action nodes x 20)
	new board encoding: Shape (#newboard nodes x 20)
	msk: Shape (#board&action nodes x #board&action) represents nodes parent/child relationships
		which defines the attention mask used in the transformer heads
	'''
	nodes,actnodes = graph_encoding.sudokuToNodes(sudoku.mat, curs_pos, action, action_val)
	benc,actenc,msk = graph_encoding.encodeNodes(nodes, actnodes)
	
	reward = runAction(action, sudoku, curs_pos)
	
	nodes,actnodes = graph_encoding.sudokuToNodes(sudoku.mat, curs_pos, action, -1) #action_val doesn't matter
	newbenc,_,_ = graph_encoding.encodeNodes(nodes, actnodes)
	
	return benc, actenc, newbenc, msk, reward


def enumerateMoves(depth, episode, possible_actions=[]): 
	if not possible_actions:
		possible_actions = range(4) # only move! 
	outlist = []
	if depth > 0: 
		for action in possible_actions:
			outlist.append(episode + [action])
			outlist = outlist + enumerateMoves(depth-1, episode + [action], possible_actions)
	return outlist

def generateActionValue(action: int, min_dist: int, max_dist: int):
	'''
	Generates an action value corresponding to the action.
	For movement actions, samples a dist unif on [min_dist, max_dist] and 
		chooses - or + direction based on the action (ex: -1 for left, +1 for right).

	min_dist: (int) Represents the min distance travelled.
	max_dist: (int) Represents the max distance travelled.
	'''
	# movement action
	if action in [Action.DOWN.value,Action.LEFT.value]:
		dist = np.random.randint(low=min_dist, high=max_dist+1)
		direction = -1
		displacement = dist * direction 
		return displacement

	if action in [Action.UP.value, Action.RIGHT.value]:
		dist = np.random.randint(low=min_dist, high=max_dist+1)
		direction = 1
		displacement = dist * direction 
		return displacement

	# guess or set note action
	if action in [Action.SET_GUESS.value, Action.UNSET_GUESS.value, Action.SET_NOTE.value,\
						Action.UNSET_NOTE.value]:
		digit = np.random.randint(1,10)
		return digit 

	# nop
	return 0
	

def enumerateBoards(puzzles, n, possible_actions=[], min_dist=1, max_dist=1): 
	'''
	Parameters:
	n: (int) Number of samples to generate
	min_dist: (int) Represents the min distance travelled.
	max_dist: (int) Represents the max distance travelled (inclusive)

	Returns:
	orig_board_enc: (tensor) Shape (N x #board nodes x 20), all the initial board encodings
	new_board_enc: (tensor) Shape (N x #board nodes x 20), all of the resulting board encodings due to actions
	action_enc: (tensor) Shape (N x #action nodes x 20), all of the episode encodings
	graph_mask: (tensor) Shape (N x #board&action nodes x #board&action nodes) all of the masks defined
		by the board and action node relations, to be used for attention head
	rewards: (tensor) Shape (N,) Rewards of each episode 

	'''
	# TODO: (JJ) update to handle multiple action episodes
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
	for i, ep in enumerate(lst): 
		puzzl = puzzles[i, :, :]
		sudoku.setMat(puzzl.numpy())
		curs_pos = torch.randint(SuN, (2,))
		action_val = generateActionValue(ep[0], min_dist, max_dist)
		
		# curs_pos = torch.randint(1, SuN-1, (2,)) # FIXME: not whole board!
		benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, ep[0], action_val)
		#benc,actenc,newbenc,msk,reward = encodeBoard(sudoku, curs_pos, ep[0], action_val)
		orig_boards.append(torch.tensor(benc))
		new_boards.append(torch.tensor(newbenc))
		actions.append(torch.tensor(actenc))
		masks.append(torch.tensor(msk))
		rewards[i] = reward
		
	orig_board_enc = torch.stack(orig_boards)
	new_board_enc = torch.stack(new_boards)
	action_enc = torch.stack(actions)
	graph_mask = torch.stack(masks)
	return orig_board_enc, new_board_enc, action_enc, graph_mask, rewards

def trainValSplit(data_matrix: torch.Tensor, num_eval=None, eval_ratio: float = 0.2):
	'''
	Split data matrix into train and val data matrices
	data_matrix: (torch.tensor) Containing rows of data
	num_eval: (int) If provided, is the number of rows in the val matrix
	'''
	num_samples = data_matrix.size(0)
	if num_samples <= 1:
		raise ValueError(f"data_matrix needs to be a tensor with more than 1 row")

	shuffled_data = data_matrix[torch.randperm(num_samples)]
	if not num_eval:
		num_eval = int(num_samples * eval_ratio)
	
	training_data = shuffled_data[:-num_eval]
	eval_data = shuffled_data[-num_eval:]
	return training_data, eval_data


class SudokuDataset(Dataset):
	'''
	Dataset where the ith element is a sample containing orig_board_enc,
		new_board_enc, action_enc, graph_mask, board_reward
	'''
	def __init__(self,orig_boards_enc, new_boards_enc, actions_enc, graph_masks, rewards):
		self.orig_boards_enc = orig_boards_enc
		self.new_boards_enc = new_boards_enc
		self.actions_enc = actions_enc
		self.graph_masks = graph_masks

		self.rewards = rewards 

	def __len__(self):
		return self.orig_boards_enc.size(0)

	def __getitem__(self, idx):
		sample = {
			'orig_board' : self.orig_boards_enc[idx],
			'new_board' : self.new_boards_enc[idx], 
			'action_enc' : self.actions_enc[idx],
			'graph_mask' : self.graph_masks[idx],
			'reward' : self.rewards[idx].item(),
		}
		return sample
			

def getDataLoaders(puzzles, num_samples, num_eval=2000):
	'''
	Returns a pytorch train and test dataloader
	'''
	data_dict = getDataDict(puzzles, num_samples, num_eval)
	train_dataset = SudokuDataset(data_dict.train_orig_board_encs, data_dict.train_new_board_encs,
										  data_dict.train_action_encs, data_dict.train_graph_masks,
										  data_dict.train_rewards)
	
	test_dataset = SudokuDataset(data_dict.test_orig_board_encs, data_dict.test_new_board_encs,
										  data_dict.test_action_encs, data_dict.test_graph_masks,
										  data_dict.test_rewards)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	return train_dataloader, test_dataloader


def getDataDict(puzzles, num_samples, num_eval=2000):
	'''
	Returns a dictionary containing training and test data
	'''
	orig_board_encs, new_board_encs, action_encs, graph_masks, rewards = enumerateBoards(puzzles, num_samples)
	print(orig_board_encs.shape, action_encs.shape, graph_masks.shape, rewards.shape)
	train_graph_masks, test_graph_masks = trainValSplit(graph_masks, num_eval)
	train_orig_board_encs, test_orig_board_encs = trainValSplit(orig_board_encs, num_eval=num_eval)
	train_new_board_encs, test_new_board_encs = trainValSplit(new_board_encs, num_eval=num_eval)
	train_action_encs, test_action_encs = trainValSplit(action_encs, num_eval=num_eval)
	train_rewards, test_rewards = trainValSplit(rewards, num_eval=num_eval)

	dataDict = {
		'train_graph_masks' : train_graph_masks,
		'test_graph_masks' : test_graph_masks,
		'train_orig_board_encs': train_orig_board_encs,
		'test_orig_board_encs': test_orig_board_encs,
		'train_new_board_encs': train_new_board_encs,
		'test_new_board_encs': test_new_board_encs,
		'train_action_encs': train_action_encs,
		'test_action_encs': test_action_encs,
		'train_rewards': train_rewards,
		'test_rewards': test_rewards
    }
	return dataDict

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

def getAttentionMasks(graph_masks, device):
	'''
	Returns a tensor of attention masks based on graph masks

	graph_masks: (Torch.tensor) Shape (num_samples x #nodes x #nodes) Graph mask based on action and board node 
	relations

	Return:
	(Torch.tensor) Shape (num_samples x #nodes x #nodes x n_heads)


	Need to repack the mask to match the attention matrix, with head duplicates. 
		Have 4h + 1 total heads. There are four categories representing relations (1:self, 2:children, 4:parents, 8:peers)
		every attention head has 4 heads, one for each relation; the heads for the children relation
		all share the same mask for example. The last mask is all-to-all 
	'''
	assert len(graph_masks.size()) == 3, f"Expect graph_masks to be shape (num_samples x #nodes x #nodes)"

	attention_masks = torch.ones((graph_masks.shape[0], graph_masks.shape[1], graph_masks.shape[2], n_heads), dtype=torch.int8)
	#attention_masks = torch.zeros((graph_masks.shape[0],graph_masks.shape[1], graph_masks.shape[2], n_heads), dtype=torch.int8) # try to save memory...

	for sample_idx in range(graph_masks.size(0)):	
		for i in range(n_heads-1): 
			j = i % 4
			attention_masks[sample_idx,:, :, i] = ( graph_masks[sample_idx] == (2**j) )
		
	# add one all-too-all mask
	if g_globalatten: 
		attention_masks[:,:,:,-1] = 1.0
	
	if g_l1atten: 
		attention_masks = torch.permute(attention_masks, (0,3,1,2)).contiguous() # L1 atten is bhts order
	# msk = msk.to_sparse() # idk if you can have views of sparse tensors.. ??
	# sparse tensors don't work with einsum, alas.
	attention_masks = attention_masks.to(device)
	return attention_masks 

def getLossMask(board_enc, device):
	'''
	TODO: (JJ) Confirm deprecation. Transformer should be able to adapt to useless zero padding
	'''
	loss_mask = torch.ones(1, board_enc.shape[1], board_enc.shape[2], device=device)
	#for i in range(11,20):
	#	loss_mask[:,:,i] *= 0.001 # semi-ignore the "latents"
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
	write_mmap(memory_dict.fd_board, pred_dict.x[0:4,:,:].cpu())
	write_mmap(memory_dict.fd_new_board, pred_dict.y[0:4,:,:].cpu())
	write_mmap(memory_dict.fd_boardp, pred_dict.yp[0:4,:,:].cpu().detach())
	write_mmap(memory_dict.fd_reward, pred_dict.reward[0:4].cpu())
	write_mmap(memory_dict.fd_rewardp, pred_dict.rp[0:4].cpu().detach())
	write_mmap(memory_dict.fd_attention, torch.stack((pred_dict.a1, pred_dict.a2), 0))
	write_mmap(memory_dict.fd_wqkv, torch.stack((pred_dict.w1, pred_dict.w2), 0))
	return 

def train(args, memory_dict, model, train_loader, optimizer, criterion, epoch):
	model.train()
	sum_batch_loss = 0.0

	for batch_idx, batch_data in enumerate(train_loader):
		(old_state, new_state, actions, graph_masks, rewards) = batch_data.to(device)
		attention_masks = getAttentionMasks(graph_masks, args.device)

		pred_data = {}
		if optimizer_name != 'psgd': 
			optimizer.zero_grad()
			new_state_pred, reward_pred, a1,a2,w1,w2 = model.forward(old_state, actions, attention_masks, epoch, None)
			pred_data = {'new_state_pred':new_state_pred, 'reward_pred': reward_pred,
							'a1':a1, 'a2':a2, 'w1':w1, 'w2':w2}
			loss = criterion(new_state_pred, new_state)
			loss.backward()
			optimizer.step() 
		else: 
			# psgd library already does loss backwards and zero grad
			def closure():
				nonlocal pred_data
				new_state_pred, reward_pred, a1,a2,w1,w2 = model.forward(old_state, actions, attention_masks, epoch, None)
				pred_data = {'new_state_pred':new_state_pred, 'reward_pred': reward_pred,
							'a1':a1, 'a2':a2, 'w1':w1, 'w2':w2}
				loss = torch.sum((new_state_pred - new_state)**2) + sum( \
					[torch.sum(1e-4 * torch.rand_like(param) * param * param) for param in model.parameters()])
				return loss
			loss = optimizer.step(closure)

		sum_batch_loss += loss.cpu().item()
		if batch_idx % 25 == 0:
			updateMemory(memory_dict, pred_data)
	
	# add epoch loss
	avg_batch_loss = sum_batch_loss / len(train_loader)
	args.fd_losslog.write(f'{epoch}\t{avg_batch_loss}\n')
	args.fd_losslog.flush()
	
def validate(args,data_dict, loss_mask, model, test_loader, criterion, epoch):
	model.eval()
	with torch.no_grad():
		test_N = test_orig_board_enc.size(0)
		for start_idx in range(0, (test_N//batch_size)*batch_size, batch_size):
			batch_idxs = torch.arange(start_idx, start_idx+batch_size)
			x = test_orig_board_enc[batch_idxs,:,:].to(device)
			a = test_action_enc[batch_idxs, :, :].to(device)
			y = test_new_board_enc[batch_idxs,:,:].to(device)
			reward = test_reward[batch_idxs]
			yp,rp,a1,a2,w1,w2 = model.forward(x,a,attn_mask, epoch, None)
			loss = criterion(yp, y)
			print('v', loss.cpu().item())
			fd_losslog.write(f'{uu}\t{loss.cpu().item()}\n')
			fd_losslog.flush()
	return 

if __name__ == '__main__':
	puzzles = torch.load('puzzles_500000.pt')
	NUM_SAMPLES = 12000
	NUM_EPOCHS = 100000
	NUM_EVAL = 2000
	device = torch.device(type='cuda', index=0)
	fd_losslog = open('losslog.txt', 'w')
	args = {"NUM_SAMPLES": NUM_SAMPLES, "NUM_EPOCHS": NUM_EPOCHS, "NUM_EVAL": NUM_EVAL,\
			 "device": device, "fd_losslog": fd_losslog}
	optimizer_name = "adam"
	
	torch.set_float32_matmul_precision('high')
	torch.manual_seed(42)
	
	# get our train and test dataloaders
	train_dataloader, test_dataloader = getDataLoaders(puzzles, args.NUM_SAMPLES, args.NUM_EVAL)
	
	# allocate memory
	memory_dict = getMemoryDict()
	
	# define model 
	model = Gracoonizer(xfrmr_dim = 20, world_dim = 20, reward_dim = 1).to(device)
	model.printParamCount()
	try: 
		model.load_checkpoint()
		print("loaded model checkpoint")
	except : 
		print("could not load model checkpoint")
	
	optimizer = getOptimizer(optimizer_name, model)
	criterion = nn.MSELoss()

	epoch_num = 0
	for _ in range(0, args.NUM_EPOCHS):
		train(args, memory_dict, model, TODOADD, optimizer, criterion, epoch_num)
		epoch_num += 1
	
	# save after training
	model.save_checkpoint()

	print("validation")

	validate(args, data_dict, loss_mask, model, test_loader, criterion, epoch)


	
	# need to allocate hidden activations for denoising
	record = []
	yp,rp,a1,a2,w1,w2 = model.forward(x, a, attn_mask, uu, record) # dummy
	denoisenet = []
	denoiseopt = []
	denoisestd = []
	hiddenl = []
	stridel = []
	for i,h in enumerate(record): 
		stride = h.shape[1]*h.shape[2]
		stridel.append(stride)
		net = NetDenoise(stride).to(device)
		opt = optim.AdamW(net.parameters(), lr=2e-4, weight_decay = 5e-2)
		denoisenet.append(net)
		denoiseopt.append(opt)
		hidden = torch.zeros(N, stride, device=device)
		hiddenl.append(hidden)
		
	print("gathering denoising data")
	# Gather denoising data on the entire dataset (excluding the few remainders modulu batch_size)
	with torch.no_grad():
		all_N = orig_board_enc.size(0)
		for start_idx in range(0, (all_N//batch_size)*batch_size, batch_size):
			batch_idxs = torch.arange(start_idx, start_idx+batch_size)
			x = orig_board_enc[batch_idxs,:,:].to(device)
			a = action_enc[batch_idxs, :, :].to(device)
			y = new_board_enc[batch_idxs,:,:].to(device)
			reward = board_reward[batch_idxs]

			record = []
			yp,rp,a1,a2,w1,w2 = model.forward(x, a, attn_mask, uu, record)
			for j,h in enumerate(record): 
				stride = stridel[j]
				hiddenl[j][batch_idxs, :] = torch.reshape(h.detach(), (batch_size, stride))
			for h in hiddenl: 
				std = torch.std(h) / 2.0
				denoisestd.append(std)
		
	
	for j,net in enumerate(denoisenet): 
		try: 
			net.load_checkpoint(f"denoise_{j}.pth")
		except: 
			print(f"could not load denoise_{j}.pth")
	
	if False: 
		print("action inference")
		for u in range(3): 
			i = torch.arange(0, batch_size) + u*batch_size
			x = board_enc[i*2,:,:].to(device)
			a = action_enc[i,:,:].to(device)
			y = board_enc[i*2+1,:,:].to(device) 
			
			ap = model.backAction(x, attn_mask, uu, y, a, loss_mask, denoisenet, denoisestd)
			
			loss = torch.sum((ap - a)**2)
			print('a', loss.cpu().item())
