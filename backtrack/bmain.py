import math
import argparse
import time
import os
import sys
import threading
import glob # for file filtering
from multiprocessing import Pool
from itertools import product
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pdb
from termcolor import colored
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
	dat = np.load(f'../satnet_both_{percent_filled}_filled_{N}.npz')
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

	np.savez(f"../satnet_enc_{percent_filled}_{N}.npz", puzzles=puzz_enc, solutions=sol_enc, coo=coo, a2a=a2a)

	return puzz_enc, sol_enc, coo, a2a
	
def encodeSudokuSteps(N, percent_filled, n_steps):
	dat = np.load(f'../satnet_both_{percent_filled}_filled_{N}.npz')
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

	np.savez(f"../satnet_{n_steps}step_enc_{percent_filled}_{N}.npz", puzzles=puzz_enc, solutions=sol_enc, coo=coo, a2a=a2a)

	return puzz_enc, sol_enc, coo, a2a

class BoardTree():
	def __init__(self, puzz, value, curs_pos, digit):
		self.puzz = puzz.astype(np.int8)
		self.value = value
		self.curs_pos = curs_pos
		self.digit = digit
		if value > 0:
			# not an end, make a list of (totally naive) possibilities.
			r,c = np.where(puzz == 0)
			indx = np.zeros((r.shape[0],2))
			indx[:,0] = r
			indx[:,1] = c
			# form the cartesian product between indx and [1..9]
			digits = np.arange(1,10)
			digits = digits[:, np.newaxis]
			indx = np.concatenate([indx.repeat(len(digits),axis=0), \
					np.tile(digits, (len(indx),1)), \
					np.zeros((len(digits)*len(indx),1)) ], axis=1)
			indx = indx.astype(np.int8) # save 8x memory
			# index is hence r,c,digit,value
			self.possible = indx
		else:
			self.possible = None # dead-end
		self.kids = [] # tuple mapping index to object

	def getKid(self, sudoku):
		u = np.where(self.possible[:,3] == 0)
		u = u[0]
		ind = np.random.randint(0,u.shape[0])
		# this step should be replaced with a policy
		# which emits a weighting over all possible moves. 
		# ind = 0
		indx = u[ind]
		r = self.possible[indx,0]
		c = self.possible[indx,1]
		d = self.possible[indx,2]
		kid_puzz = np.array(self.puzz) # deep copy
		kid_puzz[r,c] = d
		sudoku.setMat(kid_puzz)
		if sudoku.checkIfValid():
			value = 1
		else:
			value = -1
		node = BoardTree(kid_puzz, value, [r,c], d)
		self.kids.append((indx, node))
		self.possible[indx,3] = value # redundant but ok
		return node, value

	def hasPoss(self):
		# determine if there are untested kids ( value +1 or -1)
		return np.sum(self.possible[:,3] == 0) > 0
	def isDone(self):
		return np.sum(self.puzz == 0) == 0

	def solve(self, sudoku, k):
		found = False
		while self.hasPoss() and (not found) and k > 0:
			node,value = self.getKid(sudoku)
			k = k-1
			while value < 0 and self.hasPoss():
				node,value = self.getKid(sudoku)
				k = k-1
			if value > 0 and node.isDone():
				print('solution!')
				sudoku.printSudoku("", node.puzz, curs_pos=node.curs_pos)
				found = True
				return True,k
			elif value > 0:
				# found a 1-step working option, dfs!
				# print('progress:')
				# sudoku.printSudoku("", node.puzz, curs_pos=node.curs_pos)
				found,k = node.solve(sudoku,k)
		# could not find an option, backtrack.
		if not found:
			# print('backtracking from')
			# sudoku.printSudoku("", self.puzz, curs_pos=self.curs_pos)
			return False,k
		else:
			return True,k

	def count(self):
		n = len(self.kids)
		for (_,kid) in self.kids:
			n = n + kid.count()
		return n

	def flat(self, x, c, v, d, indx):
		for _,kid in self.kids:
			x[indx,:,:] = kid.puzz
			c[indx,:] = kid.curs_pos
			v[indx] = kid.value
			d[indx] = kid.digit
			indx = indx+1
		for _,kid in self.kids:
			indx = kid.flat(x,c,v,d,indx)
		return indx

	def flatten(self):
		n = self.count()
		x = np.zeros((n,9,9), dtype=np.int8)
		c = np.zeros((n,2), dtype=np.int8)
		v = np.zeros((n,), dtype=np.int8)
		d = np.zeros((n,), dtype=np.int8)
		self.flat(x,c,v,d,0)
		return x,c,v,d # board, cursor, value, digit

g_puzzles = np.zeros((9,9,9))

def singleBacktrack(j):
	global g_puzzles
	sudoku = Sudoku(9,60)
	bt = BoardTree(g_puzzles[j], 0.01, [0,0], 0)
	bt.solve(sudoku, 100)
	x,c,v,d = bt.flatten()
	benc, _, _, board_loc = encodeSudoku(x)
	mask = np.zeros(benc.shape, dtype=np.int8)
	loc = board_loc[c[0], c[1]]
	mask[loc, 10+d] = 1
	# multiply the mask by the value to get the target.
	return x,mask,v

def generateBacktrack(puzzles, N):
	global g_puzzles
	g_puzzles = puzzles
	pool = Pool() #defaults to number of available CPU's
	chunksize = 1
	results = [None for _ in range(N)]
	for ind, res in enumerate(pool.imap_unordered(singleBacktrack, range(N), chunksize)):
		results[ind] = res

	pdb.set_trace()

	x = list(map(lambda r: r[0], results))
	c = list(map(lambda r: r[1], results))
	v = list(map(lambda r: r[2], results))
	d = list(map(lambda r: r[3], results))

	x = np.concatenate(x, axis=0) # the board state
	c = np.concatenate(c, axis=0) # the cursor position
	v = np.concatenate(v, axis=0) # the value, [-1, 1]
	d = np.concatenate(d, axis=0) # the digit

	return x,c,v,d


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

	dat = np.load(f'../satnet/satnet_both_0.85_filled_{DATA_N}.npz')
	puzzles = dat['puzzles']
	puzzles = puzzles.astype(np.int8)
	sudoku = Sudoku(9,60)
	sudoku.printSudoku("",puzzles[0])
	bt = BoardTree(puzzles[0], 0.01, [0,0], 0)
	bt.solve(sudoku, 100)
	x,c,v,d = bt.flatten()

	puzzles,curs_pos,value,digit = generateBacktrack(puzzles, 1000)
	np.savez(f'satnet_backtrack_0.85.npz', \
		puzzles=puzzles, curs_pos=curs_pos, value=value, digit=digit)

	def trainValSplit(y):
		y_train = list(map(lambda x: x[:-VALID_N], y))
		y_valid = list(map(lambda x: x[-VALID_N:], y))
		y_train = torch.cat(y_train, dim=0)
		y_valid = torch.cat(y_valid, dim=0)
		return y_train, y_valid

	puzzles_train, puzzles_valid = trainValSplit(puzzles)
	values_train, values_valid = trainValSplit(values)
	curs_pos_train, curs_pos_valid = trainValSplit(curs_pos)
	digit_train, digit_valid = trainValSplit(digit)
	assert(values_train.shape[0] == puzzles_train.shape[0])
	assert(solutions_train.shape[0] == puzzles_train.shape[0])
	assert(digit_train.shape[0] == digit_train.shape[0])

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
