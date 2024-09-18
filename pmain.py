import math
import argparse
import time
import os
import sys
import glob # for file filtering
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pdb
from termcolor import colored
import matplotlib.pyplot as plt
from gracoonizer import Gracoonizer
from sudoku_gen import Sudoku
from plot_mmap import make_mmf, write_mmap
from constants import *
import sparse_encoding
from l1attn_sparse_cuda import expandCoo
# import psgd_20240912 as psgd
import psgd
import gmain
import utils


def encodeSudoku(puzz):
	nodes, _ = sparse_encoding.puzzleToNodes(puzz)
	benc, coo, a2a = sparse_encoding.encodeNodes(nodes)
	return benc, coo, a2a

def encodeSudokuAll(N):
	dat = np.load(f'satnet_both_0.75_filled_{N}.npz')
	puzzles = dat['puzzles']
	solutions = dat['solutions']
	N = puzzles.shape[0]

	puzz_enc = np.zeros((N,111,32), dtype=np.float16)
	sol_enc = np.zeros((N,111,32), dtype=np.float16)

	for i in range(N):
		puzz, coo, a2a = encodeSudoku(puzzles[i])
		sol,_,_ = encodeSudoku(solutions[i])
		# fig,axs = plt.subplots(1, 2, figsize=(12,6))
		# axs[0].imshow(puzz_enc.T)
		# axs[1].imshow(sol_enc.T)
		# plt.show()
		# print(coo)
		# print(a2a)
		puzz_enc[i,:,:] = puzz
		sol_enc[i,:,:] = sol
		if i % 1000 == 999:
			print(".", end='', flush=True)

	np.savez(f"satnet_enc_{N}.npz", puzzles=puzz_enc, solutions=sol_enc, coo=coo, a2a=a2a)

	return puzz_enc, sol_enc, coo, a2a

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train sudoku policy model")
	parser.add_argument('-a', action='store_true', help='use AdamW as the optimizer (as opposed to PSGD)')
	cmd_args = parser.parse_args()

	DATA_N = 100000
	VALID_N = DATA_N//10
	TRAIN_N = DATA_N - VALID_N
	batch_size = 64

	fname = f"satnet_enc_{DATA_N}.npz"
	try:
		file = np.load(fname)
		puzzles = file["puzzles"]
		solutions = file["solutions"]
		coo = file["coo"]
		a2a = file["a2a"]
	except Exception as error:
		print(error)
		puzzles, solutions, coo, a2a = encodeSudokuAll(DATA_N)

	puzzles = torch.from_numpy(puzzles)
	solutions = torch.from_numpy(solutions)
	coo = torch.from_numpy(coo)
	a2a = torch.from_numpy(a2a)

	indx = torch.randperm(DATA_N)
	puzzles_train = puzzles[indx[:-VALID_N],:,:]
	solutions_train = solutions[indx[:-VALID_N],:,:]
	puzzles_valid = puzzles[indx[-VALID_N:],:,:]
	solutions_valid = solutions[indx[-VALID_N:],:,:]
	print(f'loaded {fname}')

	device = torch.device('cuda:0')
	args = {"device": device}
	# use export CUDA_VISIBLE_DEVICES=1
	# to switch to another GPU
	torch.set_float32_matmul_precision('high')
	torch.backends.cuda.matmul.allow_tf32 = True

	fd_losslog = open(f'losslog_{utils.getGitCommitHash()}.txt', 'w')
	args['fd_losslog'] = fd_losslog

	model = Gracoonizer(xfrmr_dim=xfrmr_dim, world_dim=world_dim, n_heads=n_heads, n_layers=8, repeat=5, mode=0).to(device)
	model.printParamCount()

	if cmd_args.a:
		optimizer_name = "adamw"
	else:
		optimizer_name = "psgd" # adam, adamw, psgd, or sgd
	optimizer = gmain.getOptimizer(optimizer_name, model)

	hcoo = gmain.expandCoordinateVector(coo, a2a)

	for uu in range(1000):
		indx = np.random.choice(TRAIN_N, size=batch_size, replace=False)
		batch_indx = torch.from_numpy(indx)
		old_board = puzzles_train[batch_indx, :, :]
		new_board = solutions_train[batch_indx, :, :]

		old_board = torch.cat((old_board, torch.zeros_like(old_board)), dim=-1).float().to(args['device'])
		new_board = torch.cat((new_board, torch.zeros_like(new_board)), dim=-1).float().to(args['device'])

		def closure():
			new_state_preds = model.forward(old_board, hcoo)
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

	# validate!
	for j in range(VALID_N // batch_size):
		batch_indx = torch.arange(j*batch_size, (j+1)*batch_size)

		old_board = puzzles_valid[batch_indx, :, :]
		new_board = solutions_valid[batch_indx, :, :]

		old_board = torch.cat((old_board, torch.zeros_like(old_board)), dim=-1).float().to(args['device'])
		new_board = torch.cat((new_board, torch.zeros_like(new_board)), dim=-1).float().to(args['device'])

		new_state_preds = model.forward(old_board, hcoo)
		loss = torch.sum((new_state_preds[:,:,33:64] - new_board[:,:,1:32])**2)

		lloss = loss.detach().cpu().item()
		print('v',lloss)
		lloss = loss.detach().cpu().item()

		uu = uu + 1
