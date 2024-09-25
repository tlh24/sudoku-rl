import math
import argparse
import time
import os
import sys
import threading
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
import sparse_encoding
from l1attn_sparse_cuda import expandCoo
# import psgd_20240912 as psgd
import psgd
import gmain
import utils
from sudoku_gen import Sudoku


def encodeSudoku(puzz, top_node=False):
	nodes, _, board_loc = sparse_encoding.puzzleToNodes(puzz, top_node=top_node)
	benc, coo, a2a = sparse_encoding.encodeNodes(nodes)
	return benc, coo, a2a, board_loc

def encodeSudokuAll(N, percent_filled):
	dat = np.load(f'satnet_both_{percent_filled}_filled_{N}.npz')
	puzzles = dat['puzzles']
	solutions = dat['solutions']
	N = puzzles.shape[0]

	puzz_enc = np.zeros((N,111,32), dtype=np.float16)
	sol_enc = np.zeros((N,111,32), dtype=np.float16)

	for i in range(N):
		puzz, coo, a2a, _ = encodeSudoku(puzzles[i])
		sol,_,_,_ = encodeSudoku(solutions[i])
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

	np.savez(f"satnet_enc_{percent_filled}_{N}.npz", puzzles=puzz_enc, solutions=sol_enc, coo=coo, a2a=a2a)

	return puzz_enc, sol_enc, coo, a2a
	
def encodeSudokuSteps(N, percent_filled, n_steps):
	dat = np.load(f'satnet_both_{percent_filled}_filled_{N}.npz')
	puzzles = dat['puzzles']
	N = puzzles.shape[0]
	
	sudoku = Sudoku(9,60)

	puzz_enc = np.zeros((N,111,32), dtype=np.float16)
	sol_enc = np.zeros((N,111,32), dtype=np.float16)

	for i in range(N):
		puzz, coo, a2a, _ = encodeSudoku(puzzles[i])
		sudoku.setMat(puzzles[i])
		for s in range(n_steps): 
			step,_ = sudoku.takeOneStep()
			sudoku.setMat(step)
		sol,_,_,_ = encodeSudoku(step)
		puzz_enc[i,:,:] = puzz
		sol_enc[i,:,:] = sol
		# fig,axs = plt.subplots(1, 2, figsize=(12,6))
		# axs[0].imshow(puzz.T)
		# axs[1].imshow(sol.T)
		# plt.show()
		if i % 1000 == 999:
			print(".", end='', flush=True)

	np.savez(f"satnet_{n_steps}step_enc_{percent_filled}_{N}.npz", puzzles=puzz_enc, solutions=sol_enc, coo=coo, a2a=a2a)

	return puzz_enc, sol_enc, coo, a2a

def encodeSudokuValue(N, percent_filled):
	# for each puzzle, replace one clue with another digit
	# yielding two puzzles: one valid and one invalid
	# which are a minimum edit distance away.
	dat = np.load(f'satnet_both_{percent_filled}_filled_{N}.npz')
	puzzles = dat['puzzles']
	N = puzzles.shape[0]

	sudoku = Sudoku(9,60)

	puzz_enc = np.zeros((N*2,112,32), dtype=np.float16)
	value = np.zeros(N*2)

	for i in range(N):
		puzz_valid, coo, a2a, _ = encodeSudoku(puzzles[i], top_node=True)
		puzzle_invalid = np.array(puzzles[i])
		sudoku.setMat(puzzle_invalid)
		while True:
			sel = np.random.randint((2,1), 9)
			while puzzle_invalid[sel[0],sel[1]] == 0:
				sel = np.random.randint((2,1), 9)
			orig = puzzle_invalid[sel[0],sel[1]].item()
			new = np.random.randint((1,), 9).item()
			puzzle_invalid[sel[0],sel[1]] = new
			sudoku.setMat(puzzle_invalid)
			if not sudoku.checkIfValid():
				break
			else:
				puzzle_invalid[sel[0],sel[1]] = orig
		puzz_invalid, coo, a2a, _ = encodeSudoku(puzzle_invalid, top_node=True)

		if i % 1000 == 999:
			print(".", end='', flush=True)

		puzz_enc[i*2+0,:,:] = puzz_valid
		puzz_enc[i*2+1,:,:] = puzz_invalid
		value[i*2+0] = 1
		value[i*2+1] = 0

	np.savez(f"satnet_value_{percent_filled}_{N}.npz", puzzles=puzz_enc, value=value, coo=coo, a2a=a2a)

	return puzz_enc, value, coo, a2a


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train sudoku policy model")
	parser.add_argument('-a', action='store_true', help='use AdamW as the optimizer (as opposed to PSGD)')
	parser.add_argument('-c', action='store_true', help="clear, start fresh: don't load model")
	parser.add_argument('-v', action='store_true', help="train value function")
	parser.add_argument('-r', type=int, default=1, help='number of repeats or steps')
	cmd_args = parser.parse_args()

	DATA_N = 100000
	VALID_N = DATA_N//10
	batch_size = 64
	world_dim = 64
	n_steps = cmd_args.r

	puzzles = []
	solutions = []
	values = []
	# for percent_filled in [0.5,0.75]:
	for percent_filled in [0.35,0.65,0.85]:
		if cmd_args.v:
			fname = f"satnet_value_{percent_filled}_{DATA_N}.npz"
			try:
				file = np.load(fname)
				puzzles_ = file["puzzles"]
				values_ = file["value"]
				coo = file["coo"]
				a2a = file["a2a"]
				coo = torch.from_numpy(coo)
				a2a = torch.from_numpy(a2a)
			except Exception as error:
				print(error)
				puzzles_, values_, coo, a2a = encodeSudokuValue(DATA_N, percent_filled)

			puzzles_ = torch.from_numpy(puzzles_)
			values_ = torch.from_numpy(values_)
			puzzles.append(puzzles_)
			values.append(values_)
		else:
			fname = f"satnet_{n_steps}step_enc_{percent_filled}_{DATA_N}.npz"
			try:
				file = np.load(fname)
				puzzles_ = file["puzzles"]
				solutions_ = file["solutions"]
				coo = file["coo"]
				a2a = file["a2a"]
				coo = torch.from_numpy(coo)
				a2a = torch.from_numpy(a2a)
			except Exception as error:
				print(error)
				puzzles_, solutions_, coo, a2a = encodeSudokuSteps(DATA_N, percent_filled, n_steps)

			puzzles_ = torch.from_numpy(puzzles_)
			solutions_ = torch.from_numpy(solutions_)
			puzzles.append(puzzles_)
			solutions.append(solutions_)

	def trainValSplit(y):
		y_train = list(map(lambda x: x[:-VALID_N], y))
		y_valid = list(map(lambda x: x[-VALID_N:], y))
		y_train = torch.cat(y_train, dim=0)
		y_valid = torch.cat(y_valid, dim=0)
		return y_train, y_valid

	puzzles_train, puzzles_valid = trainValSplit(puzzles)
	if cmd_args.v:
		values_train, values_valid = trainValSplit(values)
		assert(values_train.shape[0] == puzzles_train.shape[0])
	else:
		solutions_train, solutions_valid = trainValSplit(solutions)
		assert(solutions_train.shape[0] == puzzles_train.shape[0])

	TRAIN_N = puzzles_train.shape[0]
	VALID_N = puzzles_valid.shape[0]
	n_tok = puzzles_train.shape[1]

	print(f'loaded {fname}; train/test {TRAIN_N} / {VALID_N}')

	device = torch.device('cuda:0')
	args = {"device": device}
	# use export CUDA_VISIBLE_DEVICES=1
	# to switch to another GPU
	torch.set_float32_matmul_precision('high')
	torch.backends.cuda.matmul.allow_tf32 = True

	fd_losslog = open(f'losslog_{utils.getGitCommitHash()}_{n_steps}.txt', 'w')
	args['fd_losslog'] = fd_losslog

	if cmd_args.v:
		model = Gracoonizer(xfrmr_dim=world_dim, world_dim=world_dim, \
			n_heads=4, n_layers=8, repeat=1, mode=0).to(device)
	else:
		model = Gracoonizer(xfrmr_dim=world_dim, world_dim=world_dim, \
			n_heads=4, n_layers=4, repeat=n_steps, mode=0).to(device)
	model.printParamCount()
	
	hcoo = gmain.expandCoordinateVector(coo, a2a)
	if not cmd_args.v:
		hcoo = hcoo[0:2] # sparse / set-layers
		hcoo.append('dense') # dense attention.
		# hcoo.insert(1, 'self')
		hcoo.append('self') # intra-token op

	if cmd_args.c: 
		print('not loading any model weights.')
	else:
		try:
			model.loadCheckpoint("checkpoints/pandaizer.pth")
			print(colored("loaded model checkpoint", "blue"))
		except Exception as error:
			print(error)

	if cmd_args.a:
		optimizer_name = "adamw"
	else:
		optimizer_name = "psgd" # adam, adamw, psgd, or sgd
	optimizer = gmain.getOptimizer(optimizer_name, model)

	input_thread = threading.Thread(target=utils.monitorInput, daemon=True)
	input_thread.start()

	bi = TRAIN_N
	for uu in range(50000):
		if bi+batch_size >= TRAIN_N:
			batch_indx = torch.randperm(TRAIN_N)
			bi = 0
		indx = batch_indx[bi:bi+batch_size]
		bi = bi + batch_size
		if cmd_args.v:
			old_board = puzzles_train[indx, :, :]
			value = values_train[indx]

			old_board = torch.cat((old_board, torch.zeros(batch_size,n_tok,world_dim-32)), dim=-1).float().to(args['device'])
			value = value.to(args['device'])

			def closure():
				value_pred = model.forward(old_board, hcoo)
				value_pred = torch.sum(value_pred[:,-1,10:20], dim=-1) # sorta arbitrary
				loss = torch.sum( (value - value_pred)**2 )\
					+ sum(\
						[torch.sum(1e-4 * \
							torch.rand_like(param) * param * param) \
							for param in model.parameters() \
						])
					# this was recommended by the psgd authors to break symmetries w a L2 norm on the weights.
				return loss
			loss = optimizer.step(closure)
		else:
			old_board = puzzles_train[indx, :, :]
			new_board = solutions_train[indx, :, :]

			old_board = torch.cat((old_board, torch.zeros(batch_size,n_tok,world_dim-32)), dim=-1).float().to(args['device'])
			new_board = torch.cat((new_board, torch.zeros(batch_size,n_tok,world_dim-32)), dim=-1).float().to(args['device'])

			def closure():
				new_state_preds = model.forward(old_board, hcoo)
				loss = torch.sum(\
						(new_state_preds[:,:,:32] - new_board[:,:,:32])**2\
						)\
					+ sum(\
						[torch.sum(1e-4 * \
							torch.rand_like(param) * param * param) \
							for param in model.parameters() \
						])
					# this was recommended by the psgd authors to break symmetries w a L2 norm on the weights.
				return loss
			loss = optimizer.step(closure)

		lloss = loss.detach().cpu().item()
		print(lloss)
		args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
		args["fd_losslog"].flush()

		if uu % 1000 == 999:
			fname = "pandaizer"
			model.saveCheckpoint(f"checkpoints/{fname}.pth")

		if utils.switch_to_validation:
			break

	# validate!
	_,_,_,board_loc = encodeSudoku(np.zeros((9,9)), \
		top_node = cmd_args.v)
	sudoku = Sudoku(9,60)
	n_valid = 0
	n_total = 0
	with torch.no_grad():
		for j in range(VALID_N // batch_size):
			batch_indx = torch.arange(j*batch_size, (j+1)*batch_size)

			if cmd_args.v:
				old_board = puzzles_train[indx, :, :]
				value = values_train[indx]

				old_board = torch.cat((old_board, torch.zeros(batch_size,n_tok,world_dim-32)), dim=-1).float().to(args['device'])
				value = value.to(args['device'])

				value_pred = model.forward(old_board, hcoo)
				value_pred = torch.sum(value_pred[:,-1,10:20], dim=-1)
				loss = torch.sum( (value - value_pred)**2 )

				n_valid = n_valid + torch.sum(\
					torch.abs(value - value_pred) < 0.4)
				n_total = n_total + batch_size
			else:
				old_board = puzzles_valid[batch_indx, :, :]
				new_board = solutions_valid[batch_indx, :, :]

				old_board = torch.cat((old_board, torch.zeros_like(old_board)), dim=-1).float().to(args['device'])
				new_board = torch.cat((new_board, torch.zeros_like(new_board)), dim=-1).float().to(args['device'])

				new_state_preds = model.forward(old_board, hcoo)
				loss = torch.sum(\
					(new_state_preds[:,:,:32] - new_board[:,:,:32])**2 \
					)
			lloss = loss.detach().cpu().item()
			print('v',lloss)
			args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
			args["fd_losslog"].flush()

			if not cmd_args.v:
				# decode and check
				for k in range(batch_size):
					benc = new_state_preds[k,:,:].squeeze().cpu().numpy()
					sol = sparse_encoding.decodeBoard(benc, board_loc)
					sudoku.setMat(sol)
					valid_cell = (sol > 0.95) * (sol < 9.05)
					complete = np.prod(valid_cell)
					if sudoku.checkIfValid() and complete > 0.5:
						n_valid = n_valid + 1
					else:
						obenc = old_board[k,:,:].squeeze().cpu().numpy()
						puz = sparse_encoding.decodeBoard(obenc, board_loc, argmax=True)
						print('failed on this puzzle:')
						sudoku.printSudoku("", puz)
						print("sol:")
						sudoku.printSudoku("", sol)
					n_total = n_total + 1

			uu = uu + 1

	print(f"Validation: vaild {n_valid} of {n_total}, {100.0*n_valid/n_total}")
