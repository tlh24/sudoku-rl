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

def runAction(action: int, sudoku, cursPos): 
	# run the action, update the world, return the reward.
	# act = b % 4
	reward = -0.05
	if action == Action.UP.value : 
		cursPos[0] -= 1
	if action == Action.RIGHT.value: 
		cursPos[1] += 1
	if action == Action.DOWN.value: 
		cursPos[0] += 1
	if action == Action.LEFT.value: 
		cursPos[1] -= 1
	cursPos[0] = cursPos[0] % SuN # wrap at the edges; 
	cursPos[1] = cursPos[1] % SuN # works for negative nums
			
	if True: 
		print(f'runAction @ {cursPos[0]},{cursPos[1]}: {action}')
	
	return reward


def encodeBoard(sudoku, cursPos, action): 
	'''
	Encodes the current board state and encodes the given action,
		runs the action, and then encodes the new board state
	
	Returns:
	board encoding, action encoding, new board encoding, 
	'''
	nodes,actnodes = graph_encoding.sudoku_to_nodes(sudoku.mat, cursPos, action)
	benc,actenc,msk = graph_encoding.encode_nodes(nodes, actnodes)
	
	reward = runAction(action, sudoku, cursPos)
	
	nodes,actnodes = graph_encoding.sudoku_to_nodes(sudoku.mat, cursPos, action)
	newbenc,_,_ = graph_encoding.encode_nodes(nodes, actnodes)
	
	return benc, actenc, newbenc, msk, reward


def enumerateMoves(depth, episode): 
	# TODO: (JJ) Change to include more moves 
	# moves = range(8)
	moves = range(4) # only move! 
	outlist = []
	if depth > 0: 
		for m in moves:
			outlist.append(episode + [m])
			outlist = outlist + enumerateMoves(depth-1, episode + [m])
	return outlist


def enumerateBoards(puzzles, n): 
	'''
	
	'''
	lst = enumerateMoves(1, [])
	if len(lst) < n: 
		rep = n // len(lst) + 1
		lst = lst * rep
	if len(lst) > n: 
		lst = random.sample(lst, n)
	sudoku = Sudoku(SuN, SuK)
	boards = [] # collapse to one tensor afterward. 
	actions = []
	rewards = torch.zeros(n)
	for i, ep in enumerate(lst): 
		puzzl = puzzles[i, :, :]
		sudoku.setMat(puzzl.numpy())
		cursPos = torch.randint(SuN, (2,))
		# cursPos = torch.randint(1, SuN-1, (2,)) # FIXME: not whole board!
		benc,actenc,newbenc,msk,reward = encodeBoard(sudoku, cursPos, ep[0])
		boards.append(torch.tensor(benc))
		boards.append(torch.tensor(newbenc))
		actions.append(torch.tensor(actenc))
		rewards[i] = reward
		
	board_enc = torch.stack(boards) # note! alternating old and new. 
	action_enc = torch.stack(actions)
	board_msk = torch.tensor(msk)
	return board_enc,action_enc,board_msk,rewards


if __name__ == '__main__':
	puzzles = torch.load('puzzles_500000.pt')
	N = 12000
	device = torch.device(type='cuda', index=0)
	torch.set_float32_matmul_precision('high')
	
	board_enc,action_enc,board_msk,board_reward = enumerateBoards(puzzles, N)
	
	print(board_enc.shape, action_enc.shape, board_msk.shape, board_reward.shape)
	
	fd_board = make_mmf("board.mmap", [batch_size, token_cnt, world_dim])
	fd_new_board = make_mmf("new_board.mmap", [batch_size, token_cnt, world_dim])
	fd_boardp = make_mmf("boardp.mmap", [batch_size, token_cnt, world_dim])
	fd_reward = make_mmf("reward.mmap", [batch_size, reward_dim])
	fd_rewardp = make_mmf("rewardp.mmap", [batch_size, reward_dim])
	fd_attention = make_mmf("attention.mmap", [2, token_cnt, token_cnt, n_heads])
	fd_wqkv = make_mmf("wqkv.mmap", [n_heads*2,2*xfrmr_dim,xfrmr_dim])

	fd_losslog = open('losslog.txt', 'w')
	
	# need to repack the mask to match the attention matrix, with head duplicates. 
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
	
	use_adamw = False
	
	if use_adamw: 
		optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 5e-2)
		
	else: 
		optimizer = psgd.LRA(model.parameters(),lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=10, grad_clip_max_norm=5)
	
	uu = 0
	beshape = board_enc.shape
	lossmask = torch.ones(batch_size,beshape[1],beshape[2], device=device)
	for i in range(11,20):
		lossmask[:,:,i] *= 0.001 # semi-ignore the "latents"

	if True:
		for u in range(100000): 
			# model.zero_grad() enabled later
			i = torch.randint(N-2000, (batch_size,)) * 2
			x = board_enc[i,:,:].to(device)
			a = action_enc[i//2,:,:].to(device)
			y = board_enc[i+1,:,:].to(device) 
			reward = board_reward[i//2]
			
			# if use_adamw: 
			# optimizer.zero_grad()
			yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, u, None)
			# yp = yp * lossmask
			# loss = torch.sum((yp - y)**2)
			# loss.backward()
			# optimizer.step() 
			# else: 
			def closure():
				yp,rp,a1,a2,w1,w2 = model.forward(x, a, msk, u, None)
				yp = yp * lossmask
				loss = torch.sum((yp - y)**2) + sum( \
                [torch.sum(1e-4 * torch.rand_like(param) * param * param) for param in model.parameters()])
				return loss
			loss = optimizer.step(closure)
            
			# print(loss.cpu().item())
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
	for u in range( (2000-batch_size*2) // (batch_size*2) ): 
		i = torch.arange(0,batch_size) + u*batch_size + (N-2000)
		i = i*2
		x = board_enc[i,:,:].to(device)
		a = action_enc[i//2,:,:].to(device)
		y = board_enc[i+1,:,:].to(device) 
		reward = board_reward[i//2]
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
	for u in range( (N - batch_size*2) // (batch_size*2) ): 
		i = torch.arange(0,batch_size) + u*batch_size
		x = board_enc[i*2,:,:].to(device)
		a = action_enc[i,:,:].to(device)
		y = board_enc[i*2+1,:,:].to(device) 
		reward = board_reward[i//2]
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
			x = board_enc[i*2,:,:].to(device)
			a = action_enc[i,:,:].to(device)
			y = board_enc[i*2+1,:,:].to(device) 
			
			ap = model.backAction(x, msk, uu, y, a, lossmask, denoisenet, denoisestd)
			
			loss = torch.sum((ap - a)**2)
			print('a', loss.cpu().item())
