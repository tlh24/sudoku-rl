import math
import random
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

def printPoss(indent, poss):
	for pi in range(27):
		print(indent, end="")
		for pj in range(27):
			i = pi // 3
			j = pj // 3
			d = (pi % 3) * 3 + pj % 3
			k = i // 3 + j // 3
			p = poss[i,j,d]
			attr = ["bold","reverse"]
			if p <= 0: 
				p = -1*p
				attr = []
			if p > 0: 
				color = "blue" if k % 2 == 0 else "red"
				light = "light_" if (i*27 + j) % 2 == 0 else ""
				print(colored(d+1, color, attrs=attr), end="")
			else:
				color = "cyan" if k % 2 == 0 else "magenta"
				light = "light_" if (i*27 + j) % 2 == 0 else ""
				print(colored('`', f"{light}{color}"), end="")
			# breaks
			if pj % 3 == 2: 
				print("", end=" ")
			if pj == 26: 
				print("")
			# 	if pi % 3 == 2: 
			# 		for s in range(9): 
			# 			print("   ", end="")
			# 		print("")


def encodeSudoku(puzz, top_node=False):
	nodes, _, board_loc = sparse_encoding.puzzleToNodes(puzz, top_node=top_node)
	benc, coo, a2a = sparse_encoding.encodeNodes(nodes)
	return benc, coo, a2a, board_loc
	
def puzz2poss(puzz): 
	# convert a 9x9 sudoku into a 9x9x9 possibility tensor
	# where 0 = unknown, -1 = impossible, 1 = certain
	poss = np.zeros((9,9,9), dtype=np.int8)
	for i in range(9): 
		for j in range(9): 
			d = puzz[i,j]
			if d > 0 and d < 10: 
				poss[i,j,:] = -1
				poss[i,j,d-1] = 1
	return poss
	
def poss2puzz(poss): 
	puzz = np.zeros((9,9), dtype=np.int8)
	for i in range(9): 
		for j in range(9):
			if np.sum(poss[i,j,:] > 0) == 1: 
				puzz[i,j] = np.argmax(poss[i,j,:]) + 1
	return puzz
	
def poss2guess(poss): 
	# pick an undefined cell, set to 1
	# sel = np.argwhere(poss == 0)[0]
	sel = random.choice( np.argwhere(poss == 0) )
	guess = np.zeros((9,9,9), dtype=np.int8)
	guess[sel[0], sel[1], sel[2]] = 1
	return guess
	
def checkValid(poss): 
	# see if a given poss tensor violates sudoku rules
	m = poss > 0
	for axis in range(3): 
		if np.max(np.sum(m, axis=axis)) > 1: 
			return False
	# block dim
	for b in range(9): 
		i = (b // 3) * 3
		j = (b % 3) * 3
		mm = np.reshape(m[i:i+3, j:j+3, :], (9,9))
		if np.max(np.sum(mm, axis=0)) > 1: 
			return False
	return True
	
def checkDone(poss): 
	m = poss > 0
	one = np.ones((9,9), dtype=np.int8)
	for axis in range(3): 
		b = np.sum(m, axis=axis) - one
		if np.sum(b) != 0: 
			return False
	return True
	
def canGuess(poss): 
	# see if we have options
	return np.sum(poss == 0) > 0
	
def solve(poss, k, sudoku, record, debug=False):
	# input is always valid. 
	guesses = np.zeros((9,9,9), dtype=np.int8)
	rem = np.zeros((9,9,9), dtype=np.int8)
	while canGuess(poss + guesses + rem) and k > 0:
		guess = poss2guess(poss + guesses + rem)
		poss2 = np.array(poss + guesses + guess) # no rem in inheritance! 
		k = k-1
		if checkValid(poss2): 
			if checkDone(poss2): 
				if debug: 
					print("solved!")
					sudoku.printSudoku("", poss2puzz(poss2))
				record.append((poss, guesses + guess))
				return True, poss2, k
			else: # not done, recurse
				if debug: 
					print("progress:")
					sel = np.argwhere(guess == 1)
					sel = sel[0]
					sudoku.printSudoku("", poss2puzz(poss2), curs_pos=sel[:2])
				rem = np.zeros((9,9,9), dtype=np.int8) # reset: different branch 
				valid,ret,k = solve(poss2, k, sudoku, record, debug)
				if valid:
					record.append((poss, guesses + guess))
					return valid, ret, k
				else:
					guesses = guesses - guess
					rem = rem - ret - guess
					if debug: 
						print("backtracking 1: removing")
						sudoku.printSudoku("", poss2puzz(-1*rem))
		else: 
			if debug: 
				print("removing guess")
			guesses = guesses - guess
		
	if debug: 
		print("backtracking 2: can't guess; poss:")
		sudoku.printSudoku("", poss2puzz(poss))
		print("rem: ")
		sudoku.printSudoku("", poss2puzz(-1*rem))
	record.append((poss, guesses))
	return False, -1*rem, k


class BoardTree():
	def __init__(self, puzz, value, curs_pos, digit, poss):
		self.puzz = puzz.astype(np.int8)
		self.value = value
		self.curs_pos = curs_pos
		self.digit = digit # already in the puzzle @ curs_pos
		if poss is None: 
			# not an end, make a list of (totally naive) possibilities.
			r,c = np.where(puzz == 0)
			indx = np.zeros((r.shape[0],2))
			indx[:,0] = r
			indx[:,1] = c
			# form the cartesian product between indx and [1..9]
			digits = np.arange(1,10)
			digits = digits[:, np.newaxis]
			indx = np.concatenate([indx.repeat(len(digits),axis=0), np.tile(digits, (len(indx),1)), np.zeros((len(digits)*len(indx),1)) ], axis=1)
			indx = indx.astype(np.int8) # save 8x memory
			# possible is hence r,c,digit,value
			np.random.shuffle(indx) # as we have no prior preference
			self.possible = indx
		else: 
			self.possible = np.array(poss) 
		self.kids = [] # tuple mapping index to object
		
	def valueKids(self, value_fn): 
		if value_fn is not None: 
			value_indx, value_digit = value_fn(self.puzz)
			# returns sorted in descending order
			self.possible = np.zeros((value_indx.shape[0],4)).astype(np.int8)
			self.possible[:,:2] = value_indx.astype(np.int8)
			self.possible[:,2] = value_digit.astype(np.int8)

	def getKid(self, sudoku, value_fn):
		assert(len(self.kids) < self.possible.shape[0])
		indx = len(self.kids) # get the next (ranked) kid
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
		# print(f"getKid {value}")
		# sudoku.printSudoku("", kid_puzz, curs_pos=[r,c])
		node = BoardTree(kid_puzz, value, [r,c], d, self.possible[indx+1:,:])
		node.valueKids(value_fn)
		self.kids.append(node)
		self.possible[indx,3] = value # redundant but ok
		return node, indx

	def hasPoss(self):
		# determine if there are untested kids ( value +1 or -1)
		return np.sum(self.possible[:,3] == 0) > 0
	def isDone(self):
		return np.sum(self.puzz == 0) == 0

	def solve(self, sudoku, k, value_fn, debug=False):
		found = False
		while self.hasPoss() and (not found) and k > 0:
			node,value = self.getKid(sudoku, value_fn)
			k = k-1
			while value < 0 and self.hasPoss():
				node,value = self.getKid(sudoku, value_fn)
				k = k-1
			if value > 0 and node.isDone():
				if debug: 
					print('solution!')
					sudoku.printSudoku("", node.puzz, curs_pos=node.curs_pos)
				else: 
					print('.', end='', flush=True)
				found = True
				return True,k
			elif value > 0:
				# found a 1-step working option, dfs!
				if debug:
					print('progress:')
					sudoku.printSudoku("", node.puzz, curs_pos=node.curs_pos)
				found,k = node.solve(sudoku, k, value_fn, debug)
		# could not find an option, backtrack.
		if not found:
			if debug:
				print('backtracking from')
				sudoku.printSudoku("", self.puzz, curs_pos=self.curs_pos)
			# if there is no solution from this node, set value accordingly
			self.value = -1
			# note: in the case that this particular guess is revisited, 
			# it will be a child of a different node & different board state.
			return False,k
		else:
			return True,k
			
	# def solve(self, sudoku, k, value_fn, debug=False): 
	# 	found = False
	# 	while self.hasPoss() and (not found) and k > 0:
	# 		node,indx = self.getKid(sudoku, value_fn)
	# 		k = k-1
	# 		k,value,backtrack = node.solve(sudoku, k, value_fn, debug)
	# 		self.possibilites[indx,3] = value
	# 		if value > 0 and node.isDone():
	# 			if debug: 
	# 				print('solution!')
	# 				sudoku.printSudoku("", node.puzz, curs_pos=node.curs_pos)
	# 			else: 
	# 				print('.', end='', flush=True)
	# 			found = True
	# 			return k, 1, []
	# 		else
	# 	if found:
	# 		return k, 1, []
	# 	else:
	# 		if debug:
	# 			print('backtracking from')
	# 			sudoku.printSudoku("", self.puzz, curs_pos=self.curs_pos)
	# 		return k, -1, self.kids
			

	def count(self):
		n = len(self.kids)
		for kid in self.kids:
			n = n + kid.count()
		return n

	def flat(self, x, c, v, d, indx):
		for kid in self.kids:
			x[indx,:,:] = kid.puzz
			c[indx,:] = kid.curs_pos
			v[indx] = kid.value
			d[indx] = kid.digit
			indx = indx+1
		for kid in self.kids:
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

	batch_size = 64
	world_dim = 64
	n_steps = cmd_args.r

	dat = np.load(f'../satnet/satnet_both_0.85_filled_100000.npz')
	puzzles = dat["puzzles"]
	sudoku = Sudoku(9,60)

	npz_file = f'satnet_backtrack_0.85.npz'
	try:
		file = np.load(npz_file)
		poss_all = file["poss_all"]
		guess_all = file["guess_all"]
		print(f"number of supervised examples: {poss_all.shape[0]}")
	except Exception as error:
		record = []
		for i in range(16000): 
			_,sol,_ = solve(puzz2poss(puzzles[i]), 256, sudoku, record, False)
			if i % 100 == 99: 
				print(".", end="", flush=True)
			# print("solve() result:")
			# sudoku.printSudoku("", poss2puzz(sol))
		
		n = len(record)
		print(f"number of supervised examples: {n}")
		poss_all = np.zeros((n,9,9,9), dtype=np.int8)
		guess_all = np.zeros((n,9,9,9), dtype=np.int8)
		for i,(poss,guess) in enumerate(record):
			# sudoku.printSudoku("r ",poss2puzz(poss))
			# printPoss("", guess)
			poss_all[i] = poss
			guess_all[i] = guess
		np.savez(npz_file, poss_all=poss_all, guess_all=guess_all)
	
	DATA_N = poss_all.shape[0]
	VALID_N = DATA_N//10

	def trainValSplit(y):
		y_train = torch.tensor(y[:-VALID_N])
		y_valid = torch.tensor(y[-VALID_N:])
		return y_train, y_valid

	poss_train, poss_valid = trainValSplit(poss_all)
	guess_train, guess_valid = trainValSplit(guess_all)
	assert(guess_train.shape[0] == poss_train.shape[0])

	TRAIN_N = poss_train.shape[0]
	VALID_N = poss_valid.shape[0]

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
	
	if cmd_args.c: 
		print('not loading any model weights.')
	else:
		try:
			model.loadCheckpoint("checkpoints/pandaizer.pth")
			print(colored("loaded model checkpoint", "blue"))
		except Exception as error:
			print(error)
			
	# null encoding that will be replaced. 
	benc, coo, a2a, board_loc = encodeSudoku(puzzles[0])
	n_tok = benc.shape[0]
	benc = torch.from_numpy(benc).float()
	benc = torch.tile(benc.unsqueeze(0), (batch_size,1,1) )
	benc = torch.cat((benc, torch.zeros(batch_size,n_tok,world_dim-32)), dim=-1)
	benc = benc.to(device)
	benc_new = benc.clone()
	benc_mask = torch.zeros_like(benc)
	# the meat of the ecoding will be replaced by (clipped) poss. 
	
	hcoo = gmain.expandCoordinateVector(coo, a2a)
	hcoo = hcoo[0:2] # sparse / set-layers
	hcoo.append('self') # intra-token op
	
# 	# make a new value-based BoardTree
# 	def valueFn(puzz): 
# 		benc = benc_train[0].clone()
# 		benc[30:, 10:20] = 0 # erase the old encoding
# 		puzz_flat = torch.from_numpy(np.reshape(puzz, (81,)))
# 		puzz_flat = puzz_flat.int()
# 		m = torch.arange(81)
# 		benc[m+30, puzz_flat[m]+10] = 1
# 		possible = torch.zeros(81, 10)
# 		indx = torch.where(puzz_flat == 0)
# 		possible[indx[0], 1:] = 1
# 		with torch.no_grad(): 
# 			benc_ = torch.cat((benc, torch.zeros(n_tok,world_dim-32)), dim=-1).float().to(args['device'])
# 			benc_ = benc_.unsqueeze(0) # leading batch dim
# 			preds = model.forward(benc_, hcoo)
# 		preds = preds.detach().cpu().squeeze()
# 		preds = preds[30:, 10:20]
# 		preds = (np.clip(preds, -1, 1) + 1)/2
# 		preds = preds * possible
# 		_, glob_indx = torch.sort(preds.flatten(), descending=True)
# 		sorted_coords = torch.unravel_index(glob_indx, preds.shape)
# 		# print(puzz)
# 		# plt.rcParams['toolbar'] = 'toolbar2'
# 		# fig,axs = plt.subplots(1,2,figsize=(12,6))
# 		# axs[0].imshow(benc.numpy().T)
# 		# axs[1].imshow(preds.numpy().T)
# 		# plt.show()
# 		# pdb.set_trace()
# 		# for i in range(3): 
# 		# 	row = sorted_coords[0][i] // 9
# 		# 	col = sorted_coords[0][i] % 9
# 		# 	print(row, col, sorted_coords[1][i])
# 		n = int(torch.sum(possible).item())
# 		c = torch.zeros((n,2), dtype=torch.int8)
# 		d = torch.zeros((n,), dtype=torch.int8)
# 		c[:,0] = sorted_coords[0][:n] // 9
# 		c[:,1] = sorted_coords[0][:n] % 9
# 		d = sorted_coords[1][:n]
# 		return c.numpy(), d.numpy()
# 		
# 	stats = []
# 	for i in range(2,3): 
# 		sudoku.printSudoku("", puzzles[i])
# 		bt = BoardTree(puzzles[i], 0.1, [0,0], 0)
# 		bt.valueKids(valueFn)
# 		solved,k = bt.solve(sudoku, 300, None, debug=True)
# 		stats.append((solved,300-k))
# 	print(stats)
# 	exit()

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
			batch_indx = np.random.permutation(TRAIN_N)
			bi = 0
		indx = batch_indx[bi:bi+batch_size]
		bi = bi + batch_size
		
		poss = poss_train[indx, :, :, :].reshape((batch_size,81,9))
		poss = torch.clip(poss, 0, 1)
		guess = guess_train[indx, :, :, :].reshape((batch_size,81,9))
		mask = guess != 0

		poss = poss.float().to(device)
		guess = guess.float().to(device)
		mask = mask.float().to(device)
		
		benc[:,-81:,11:20] = poss
		benc_new[:,-81:,11:20] = guess # is signed
		benc_mask[:,-81:,11:20] = mask

		def closure():
			benc_pred = model.forward(benc, hcoo)
			global pred_data
			pred_data = {'old_board':benc, \
				'new_board':benc_new, \
				'new_state_preds':benc_pred,\
				'rewards':None, 'reward_preds':None,'w1':None, 'w2':None}
			
			loss = torch.sum( ((benc_pred - benc_new)*benc_mask)**2 ) \
				+ sum(\
					[torch.sum(1e-5 * \
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
			global pred_data
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

		# # linear psgd warm-up
		# if not cmd_args.a:
		# 	if uu < 5000:
		# 		optimizer.lr_params = \
		# 			0.0075 * (uu / 5000) / math.pow(n_steps, 1.0)
		# 			# made that scaling up
					
		if uu % 25 == 0:
			# print(prof.key_averages( group_by_input_shape=True ).table( sort_by="cuda_time_total", row_limit=50))
			gmain.updateMemory(memory_dict, pred_data)

		if utils.switch_to_validation:
			break

	# validate!
	
	with torch.no_grad():
		for j in range(VALID_N // batch_size):
			indx = torch.arange(j*batch_size, (j+1)*batch_size)
			
			poss = poss_valid[indx, :, :, :].reshape((batch_size,81,9))
			poss = np.clip(poss, 0, 1)
			guess = guess_valid[indx, :, :, :].reshape((batch_size,81,9))
			mask = guess != 0

			poss = poss.float().to(device)
			guess = guess.float().to(device)
			mask = mask.float().to(device)
			
			benc[:,-81:,11:20] = poss
			benc_new[:,-81:,11:20] = guess # is signed
			benc_mask[:,-81:,11:20] = mask
			
			benc_pred = model.forward(benc, hcoo)
			
			loss = torch.sum( ((benc_pred - benc_new)*benc_mask)**2 )
			
			lloss = loss.detach().cpu().item()
			print('v',lloss)
			args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
			args["fd_losslog"].flush()

			uu = uu + 1
