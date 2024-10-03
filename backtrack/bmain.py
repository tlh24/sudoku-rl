import math
import argparse
import time
import os
import sys
import threading
import resource
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

class BoardTree():
	def __init__(self, puzz, value, curs_pos, digit):
		self.puzz = puzz.astype(np.int8)
		self.value = value
		self.curs_pos = curs_pos
		self.digit = digit # already in the puzzle @ curs_pos
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
				# print('solution!')
				print('.', end='', flush=True)
				# sudoku.printSudoku("", node.puzz, curs_pos=node.curs_pos)
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
	n_rollouts = 100
	sudoku = Sudoku(9,60)
	bt = BoardTree(g_puzzles[j], 0.01, [0,0], 0)
	bt.solve(sudoku, n_rollouts)
	x,c,v,d = bt.flatten()
	benc = []
	mask = []
	for i in range(x.shape[0]): 
		benc_, coo, a2a, board_loc = encodeSudoku(x[i])
		mask_ = np.zeros(benc_.shape, dtype=np.int8)
		loc = board_loc[c[i,0], c[i,1]]
		mask_[loc, 10+d[i]] = 1
		benc.append(benc_.astype(np.int8))
		mask.append(mask_)
		# multiply the mask by the value to get the target.
	benc = np.stack(benc) # board encodings
	mask = np.stack(mask) # masked value of given positions
	return x,c,v,d,benc,mask,coo,a2a

def generateBacktrack(puzzles, N):
	global g_puzzles
	g_puzzles = puzzles
	pool = Pool() #defaults to number of available CPU's
	chunksize = 16
	results = [None for _ in range(N)]
	for ind, res in enumerate(pool.imap_unordered(singleBacktrack, range(N), chunksize)):
	# for ind in range(N): # debug
	# 	res = singleBacktrack(ind)
		results[ind] = res

	x = list(map(lambda r: r[0], results))
	value = list(map(lambda r: r[2], results))
	benc = list(map(lambda r: r[4], results))
	mask = list(map(lambda r: r[5], results))
	coo = results[0][6]
	a2a = results[0][7]
	
	puzzles = np.concatenate(x)
	benc = np.concatenate(benc)
	mask = np.concatenate(mask)
	value = np.concatenate(value)

	return puzzles, benc, mask, value, coo, a2a


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train sudoku policy model")
	parser.add_argument('-a', action='store_true', help='use AdamW as the optimizer (as opposed to PSGD)')
	parser.add_argument('-c', action='store_true', help="clear, start fresh: don't load model")
	parser.add_argument('-v', action='store_true', help="train value function")
	parser.add_argument('-r', type=int, default=1, help='number of repeats or steps')
	parser.add_argument('--no-train', action='store_true', help="don't train the model.")
	cmd_args = parser.parse_args()
	
	# increase the number of file descriptors for multiprocessing
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (8*2048, rlimit[1]))

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

	npz_file = f'satnet_backtrack_0.85.npz'
	try:
		file = np.load(npz_file)
		benc = file["benc"]
		mask = file["mask"]
		value = file["value"]
		coo = file["coo"]
		a2a = file["a2a"]
		coo = torch.from_numpy(coo)
	except Exception as error:
		print(error)
		puzzles,benc,mask,value,coo,a2a = generateBacktrack(puzzles, 8000)
		np.savez(npz_file, \
			puzzles=puzzles, benc=benc, mask=mask, value=value, coo=coo, a2a=a2a)

	def trainValSplit(y):
		y_train = torch.tensor(y[:-VALID_N])
		y_valid = torch.tensor(y[-VALID_N:])
		return y_train, y_valid

	benc_train, benc_valid = trainValSplit(benc)
	mask_train, mask_valid = trainValSplit(mask)
	value_train, value_valid = trainValSplit(value)
	assert(mask_train.shape[0] == benc_train.shape[0])
	assert(value_train.shape[0] == benc_train.shape[0])

	TRAIN_N = benc_train.shape[0]
	VALID_N = benc_valid.shape[0]
	n_tok = benc_train.shape[1]

	print(f'loaded {npz_file}; train/test {TRAIN_N} / {VALID_N}')

	device = torch.device('cuda:0')
	args = {"device": device}
	# use export CUDA_VISIBLE_DEVICES=1
	# to switch to another GPU
	torch.set_float32_matmul_precision('high')
	torch.backends.cuda.matmul.allow_tf32 = True

	fd_losslog = open(f'losslog_{utils.getGitCommitHash()}_{n_steps}.txt', 'w')
	args['fd_losslog'] = fd_losslog
	
	memory_dict = gmain.getMemoryDict("../")

	model = Gracoonizer(xfrmr_dim=world_dim, world_dim=world_dim, \
			n_heads=8, n_layers=9, repeat=n_steps, mode=0).to(device)
	model.printParamCount()
	
	hcoo = gmain.expandCoordinateVector(coo, a2a)
	hcoo = hcoo[0:2] # sparse / set-layers
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
	avg_duration = 0.0
	
	for uu in range(200000):
		time_start = time.time()
		if bi+batch_size >= TRAIN_N:
			batch_indx = torch.randperm(TRAIN_N)
			bi = 0
		indx = batch_indx[bi:bi+batch_size]
		bi = bi + batch_size
		
		old_board = benc_train[indx, :, :]
		mask = mask_train[indx, :, :]
		value = value_train[indx]

		old_board = torch.cat((old_board, torch.zeros(batch_size,n_tok,world_dim-32)), dim=-1).float().to(args['device'])
		mask = mask.float().to(args['device'])
		value = value.float().to(args['device'])

		def closure():
			new_state_preds = model.forward(old_board, hcoo)
			global pred_data
			pred_data = {'old_board':old_board, \
				'new_board':mask*value[:,None,None], 'new_state_preds':new_state_preds,\
				'rewards':None, 'reward_preds':None,'w1':None, 'w2':None}
			
			loss = torch.sum(\
				(torch.sum(new_state_preds[:,:,:32] * mask, dim=[1,2]) \
					- value)**2 )
			+ sum(\
				[torch.sum(1e-6 * \
					torch.rand_like(param) * param * param) \
					for param in model.parameters() \
				])
				# this was recommended by the psgd authors to break symmetries w a L2 norm on the weights.
			return loss

		# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
			# with record_function("model_training"):
		if not cmd_args.no_train:
			if not cmd_args.a:
				loss = optimizer.step(closure)
			else:
				optimizer.zero_grad()
				loss = closure()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
				loss.backward()
				optimizer.step()
		else:
			with torch.no_grad():
				loss = closure()

		if uu % 25 == 0:
			# print(prof.key_averages( group_by_input_shape=True ).table( sort_by="cuda_time_total", row_limit=50))
			gmain.updateMemory(memory_dict, pred_data)

		duration = time.time() - time_start
		avg_duration = 0.99 * avg_duration + 0.01 * duration

		lloss = loss.detach().cpu().item()
		print(lloss, "\t", avg_duration)
		args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
		args["fd_losslog"].flush()

		if not cmd_args.no_train:
			if uu % 1000 == 999:
				fname = "pandaizer"
				model.saveCheckpoint(f"checkpoints/{fname}.pth")

		# linear psgd warm-up
		if not cmd_args.a:
			if uu < 5000:
				optimizer.lr_params = \
					0.0075 * (uu / 5000) / math.pow(n_steps, 1.0)
					# made that scaling up
					
		if uu % 25 == 0:
			# print(prof.key_averages( group_by_input_shape=True ).table( sort_by="cuda_time_total", row_limit=50))
			gmain.updateMemory(memory_dict, pred_data)

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
			indx = torch.arange(j*batch_size, (j+1)*batch_size)

			old_board = benc_valid[indx, :, :]
			mask = mask_valid[indx, :, :]
			value = value_valid[indx]

			old_board = torch.cat((old_board, torch.zeros(batch_size,n_tok,world_dim-32)), dim=-1).float().to(args['device'])
			mask = mask.float().to(args['device'])
			value = value.float().to(args['device'])

			new_state_preds = model.forward(old_board, hcoo)
			
			loss = torch.sum(\
				(torch.sum(new_state_preds[:,:,:32] * mask, dim=[1,2]) \
					- value)**2 )
			
			lloss = loss.detach().cpu().item()
			print('v',lloss)
			args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
			args["fd_losslog"].flush()

			uu = uu + 1
