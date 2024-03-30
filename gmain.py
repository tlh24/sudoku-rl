import math
import random
import numpy as np
import torch
from torch import nn, optim
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
	curs_pos[0] = curs_pos[0] % SuN # wrap at the edges; 
	curs_pos[1] = curs_pos[1] % SuN # works for negative nums
	
	if action == Action.SET_GUESS.value:
		clue = sudoku.mat[cursPos[0], cursPos[1]]
		curr = guess_mat[cursPos[0], cursPos[1]]
		if clue == 0 and curr == 0 and sudoku.checkIfSafe(curs_pos[0], curs_pos[1], num):
			# updateNotes(cursPos, num, notes)
			reward = 1
			guess_mat[cursPos[0], cursPos[1]] = num
		else:
			reward = -1
	if action == Action.UNSET_GUESS.value:
		curr = guess_mat[cursPos[0], cursPos[1]]
		if curr != 0: 
			guess_mat[cursPos[0], cursPos[1]] = 0
		else:
			reward = -0.25
			
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
	
	reward = runAction(action, action_val, sudoku, curs_pos)
	
	new_curs_enc = curs_enc + action_enc  
	
	return curs_enc, action_enc, new_curs_enc, mask, reward

	
def encodeBoard(sudoku, guess_mat, curs_pos, action, action_val):  
	'''
	Encodes the current board state and encodes the given action,
		runs the action, and then encodes the new board state
	
	Returns:
	board encoding, action encoding, new board encoding
	'''
	nodes,actnodes = graph_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, action, action_val)
	benc,actenc,msk = graph_encoding.encodeNodes(nodes, actnodes)
	
	reward = runAction(sudoku, guess_mat, curs_pos, action, action_val)
	
	nodes,actnodes = graph_encoding.sudokuToNodes(sudoku.mat, guess_mat, curs_pos, action, -1) # action_val doesn't matter
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


def enumerateBoards(puzzles, n, possible_actions=[], min_dist=1, max_dist=1): 
	'''
	Parameters:
	n: (int) Number of samples to generate
	min_dist: (int) Represents the min distance travelled.
	max_dist: (int) Represents the max distance travelled (inclusive)

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
	rewards = torch.zeros(n)
	guess_mat = np.zeros((SuN, SuN))
	for i, ep in enumerate(lst): 
		puzzl = puzzles[i, :, :]
		sudoku.setMat(puzzl.numpy())
		curs_pos = torch.randint(SuN, (2,))
		action_val = generateActionValue(ep[0], min_dist, max_dist)
		
		# curs_pos = torch.randint(1, SuN-1, (2,)) # FIXME: not whole board!
		# benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, ep[0], action_val)
		benc,actenc,newbenc,msk,reward = encodeBoard(sudoku, guess_mat, curs_pos, ep[0], action_val)
		orig_boards.append(torch.tensor(benc))
		new_boards.append(torch.tensor(newbenc))
		actions.append(torch.tensor(actenc))
		rewards[i] = reward
		
	orig_board_enc = torch.stack(orig_boards)
	new_board_enc = torch.stack(new_boards)
	action_enc = torch.stack(actions)
	board_msk = torch.tensor(msk)
	return orig_board_enc, new_board_enc, action_enc,board_msk,rewards


if __name__ == '__main__':
	puzzles = torch.load('puzzles_500000.pt')
	N = 12000
	device = torch.device(type='cuda', index=0)
	torch.set_float32_matmul_precision('high')
	
	orig_board_enc,new_board_enc,action_enc,board_msk,board_reward = enumerateBoards(puzzles, N)
	
	print(orig_board_enc.shape, new_board_enc.shape, action_enc.shape, board_msk.shape, board_reward.shape)
	
	fd_board = make_mmf("board.mmap", [batch_size, token_cnt, world_dim])
	fd_new_board = make_mmf("new_board.mmap", [batch_size, token_cnt, world_dim])
	fd_boardp = make_mmf("boardp.mmap", [batch_size, token_cnt, world_dim])
	fd_reward = make_mmf("reward.mmap", [batch_size, reward_dim])
	fd_rewardp = make_mmf("rewardp.mmap", [batch_size, reward_dim])
	fd_attention = make_mmf("attention.mmap", [2, token_cnt, token_cnt, n_heads])
	fd_wqkv = make_mmf("wqkv.mmap", [n_heads*2,2*xfrmr_dim,xfrmr_dim])

	fd_losslog = open('losslog.txt', 'w')
	
	# need to repack the mask to match the attention matrix, with head duplicates. 
	# Have 4h + 1 total heads. There are four categories representing relations (1:self, 2:children, 4:parents, 8:peers)
		# every attention head has 4 heads, one for each relation; the heads for the children relation
		# all share the same mask for example. The last mask is all-to-all 

	msk = torch.zeros((board_msk.shape[0], board_msk.shape[1], n_heads), dtype=torch.int8) # try to save memory...
	for i in range(n_heads-1): 
		j = i % 4
		msk[:, :, i] = ( board_msk == (2**j) )
	# add one all-too-all mask
	if g_globalatten: 
		msk[:,:,-1] = 1.0
	msk = msk.unsqueeze(0).expand([batch_size, -1, -1, -1])
	if g_l1atten: 
		msk = torch.permute(msk, (0,3,1,2)).contiguous() # L1 atten is bhts order
	# msk = msk.to_sparse() # idk if you can have views of sparse tensors.. ??
	# sparse tensors don't work with einsum, alas.
	msk = msk.to(device)
	
	model = Gracoonizer(xfrmr_dim = 20, world_dim = 20, reward_dim = 1).to(device)
	model.printParamCount()
	try: 
		model.load_checkpoint()
		print("loaded model checkpoint")
	except : 
		print("could not load model checkpoint")
	# torch.autograd.set_detect_anomaly(True)
	# model_opt = torch.compile(model)
	
	use_adamw = True
	
	if use_adamw: 
		optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay = 5e-2)
		
	else: 
		optimizer = psgd.LRA(model.parameters(),lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=10, grad_clip_max_norm=5)
	
	uu = 0
	beshape = orig_board_enc.shape
	lossmask = torch.ones(batch_size,beshape[1],beshape[2], device=device)
	for i in range(11,20):
		lossmask[:,:,i] *= 0.001 # semi-ignore the "latents"

	if True:
		for u in range(100000): 
			# model.zero_grad() enabled later
			i = torch.randint(N-2000, (batch_size,))
			x = orig_board_enc[i,:,:].to(device)
			a = action_enc[i,:,:].to(device)
			y = new_board_enc[i,:,:].to(device) 
			reward = board_reward[i]
			
			
			def closure():
				yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, u, None)
				yp = yp * lossmask
				loss = torch.sum((yp - y)**2) + sum( \
                [torch.sum(1e-4 * torch.rand_like(param) * param * param) for param in model.parameters()])
				return loss
			
			if use_adamw: 
				optimizer.zero_grad()
				yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, u, None)
				yp = yp * lossmask
				loss = torch.sum((yp - y)**2)
				loss.backward()
				optimizer.step() 
			else:
				loss = optimizer.step(closure)
				if u % 25 == 0: 
					yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, u, None)
            
			print(loss.cpu().item())
			fd_losslog.write(f'{uu}\t{loss.cpu().item()}\n')
			fd_losslog.flush()
			uu = uu+1
			
			if u % 25 == 0: 
				write_mmap(fd_board, x[0:4,:,:].cpu())
				write_mmap(fd_new_board, y[0:4,:,:].cpu())
				write_mmap(fd_boardp, yp[0:4,:,:].cpu().detach())
				write_mmap(fd_reward, reward[0:4].cpu())
				write_mmap(fd_rewardp, rp[0:4].cpu().detach())
				write_mmap(fd_attention, torch.stack((a1, a2), 0))
				write_mmap(fd_wqkv, torch.stack((w1, w2), 0))
		model.save_checkpoint()
	
	print("validation")
	for u in range( (2000-batch_size) // batch_size ): 
		i = torch.arange(0,batch_size) + u*batch_size + (N-2000)
		x = orig_board_enc[i,:,:].to(device)
		a = action_enc[i,:,:].to(device)
		y = new_board_enc[i,:,:].to(device) 
		reward = board_reward[i]
		yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, uu, None)
		
		# yp = yp - 4.5 * (yp > 4.5)
		# yp = yp + 4.5 * (yp < -4.5)
		# yp = torch.remainder(yp + 4.5, 9) - 4.5 # this diverges.
		yp = yp * lossmask
		
		loss = torch.sum((yp - y)**2)
		print('v', loss.cpu().item())
		fd_losslog.write(f'{uu}\t{loss.cpu().item()}\n')
		fd_losslog.flush()
		uu = uu+1
	
	# need to allocate hidden activations for denoising
	record = []
	yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, uu, record) # dummy
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
	for u in range( (N - batch_size) // batch_size ): 
		i = torch.arange(0,batch_size) + u*batch_size
		x = orig_board_enc[i,:,:].to(device)
		a = action_enc[i,:,:].to(device)
		y = new_board_enc[i,:,:].to(device) 
		reward = board_reward[i]
		record = []
		yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, uu, record)
		for j,h in enumerate(record): 
			stride = stridel[j]
			i = torch.arange(0,batch_size) + u*batch_size
			hiddenl[j][i, :] = torch.reshape(h.detach(), (batch_size, stride))
		for h in hiddenl: 
			std = torch.std(h) / 2.0
			denoisestd.append(std)
	
	for j,net in enumerate(denoisenet): 
		try: 
			net.load_checkpoint(f"denoise_{j}.pth")
		except: 
			print(f"could not load denoise_{j}.pth")
	
	if True: 
		print("action inference")
		for u in range(3): 
			i = torch.arange(0, batch_size) + u*batch_size
			x = orig_board_enc[i,:,:].to(device)
			a = action_enc[i,:,:].to(device)
			y = new_board_enc[i,:,:].to(device) 
			
			ap = model.backAction(x, msk, uu, y, a, lossmask, denoisenet, denoisestd)
			
			loss = torch.sum((ap - a)**2)
			print('a', loss.cpu().item())
