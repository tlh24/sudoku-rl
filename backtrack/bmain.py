import math
import random
import argparse
import time
import os
import sys
import resource
import glob # for file filtering
import multiprocessing as mp
import threading
from queue import Empty
from typing import List, Tuple, Any, Callable
from dataclasses import dataclass
import csv # for reading rrn
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

	
def checkIfValid(mat): 
	# verify that the current puzzle has no contradictions. 
	# this does not check if there are cells with no options - 
	# for that, use checkValid(poss)
	valid = True
	for i in range(9): 
		match = mat == i+1
		if np.max(np.sum(match, 1)) > 1: 
			valid = False
		if np.max(np.sum(match, 0)) > 1: 
			valid = False
		blocks = []
		for i in range(0, 9, 3):
			for j in range(0, 9, 3):
				block = match[i:i+3, j:j+3].flatten()
				blocks.append(block)
		match = np.array(blocks)
		if np.max(np.sum(match, 1)) > 1: 
			valid = False
	return valid

def printSudoku(indent, puzzl_mat, curs_pos=None):
	for i in range(9):
		print(indent, end="")
		for j in range(9):
			k = i // 3 + j // 3
			color = "black" if k % 2 == 0 else "red"
			p = int(puzzl_mat[i,j])
			bgcol = None
			if curs_pos is not None: 
				if int(curs_pos[0]) == i and int(curs_pos[1]) == j:
					bgcol = "on_light_yellow"
			if p == 0:
				if color == "black":
					color = "light_grey"
				if color == "red":
					color = "light_red"
				if color == "blue":
					color = "light_blue"
				if color == "magenta":
					color = "light_magenta"
			if bgcol is not None: 
				print(colored(p, color, bgcol), end=" ")
			else: 
				print(colored(p, color), end=" ")
		print()
	print(f"{indent}Valid:", checkIfValid(puzzl_mat))
	# print the string form
	for i in range(9):
		for j in range(9):
			print(puzzl_mat[i,j], end="")
	print(",")

def printPoss(indent, poss):
	print("    ", end="")
	for pj in range(9): 
		print(f"-{pj}- ", end="")
	print("")
	for pi in range(27):
		print(indent, end="")
		print(f"{pi // 3} - ", end="")
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
	
def puzz2poss(puzz): 
	# convert a 9x9 sudoku into a 9x9x9 possibility tensor
	# where 0 = unknown, -1 = impossible, 1 = guess, 2 = clue
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
	
def poss2guessDumb(poss, value_fn): 
	# pick an undefined cell, set to 1
	if value_fn and False:
		val = value_fn( np.clip(poss, 0, 1)) # ignore guesses
		val = val * (poss == 0)
		val = val - 1e6*( poss != 0 )
		flat_sel = np.argmax( val )
		sel = np.unravel_index(flat_sel, poss.shape)
	else: 
		# sel = random.choice( np.argwhere(poss == 0) )
		# random sel does not work!!! 
		# blows up the search tree! 
		sel = np.argwhere(poss == 0)[0]
	guess = np.zeros((9,9,9), dtype=np.int8)
	guess[sel[0], sel[1], sel[2]] = 1
	return guess
	
def sampleCategorical(categorical_probs):
	# adapted from SEDD catsample.py
	gumbel_norm = 1e-10 - np.log(np.random.random_sample(categorical_probs.shape) + 1e-10)
	sel_flat = np.argmax(categorical_probs / gumbel_norm)
	sel = np.unravel_index(sel_flat, categorical_probs.shape)
	return sel
	
def boxPermutation(): 
	# return an index vector that pemutes r,c
	# to box_n, box_i
	# - example: 
	# poss_box = poss[permute[:,0], permute[:,1], :].reshape((9,9,9))
	# - then sum over axis 1 (e.g.)
	# - to reverse: 
	# poss_ = poss_box[unpermute[:,0], unpermute[:,1], :].reshape((9,9,9))
	permute = np.zeros((81,2), dtype=int)
	unpermute = np.zeros((81,2), dtype=int)
	for bn in range(9): 
		for bi in range(9): 
			i = (bn // 3) * 3
			j = (bn % 3) * 3
			ii = bi // 3
			jj = bi % 3
			permute[bn*9+bi,:] = [i+ii, j+jj]
			unpermute[(i+ii)*9+(j+jj),:] = [bn, bi]
	return permute,unpermute
	
box_permute, box_unpermute = boxPermutation()

def poss2guessSmart(poss, value_fn):
	# pick the most defined cell on the board
	# (cell with the fewest number of possibilities --
	#   hence the smallest branching factor)
	# this is effectively the hidden / unhidden singles strategy.
	m = poss == 0
	n = poss > 0
	gmin = 100
	for axis in range(3): 
		s = np.sum( m, axis=axis ) + 10*np.sum( n, axis=axis)
		flat = np.argmin( s )
		a,b = np.unravel_index(flat, s.shape)
		a,b = a.item(), b.item()
		smin = s[a,b].item()
		if axis == 0 and smin < gmin: 
			c,d = a,b
			r = np.argmax( m[:,c,d] ).item()
			gmin = smin
		if axis == 1 and smin < gmin: 
			r,d = a,b
			c = np.argmax( m[r,:,d] ).item()
			gmin = smin
		if axis == 2 and smin < gmin: 
			r,c = a,b
			d = np.argmax( m[r,c,:] ).item()
		# should add in a 'box' axis via the permutation... maybe later.
	guess = np.zeros((9,9,9), dtype=np.int8)
	guess[r, c, d] = 1
	if gmin < 1:
		pdb.set_trace() # that's an error!
	return guess
	
def poss2guessRand(poss, value_fn, cntr): 
	''' for use with stochasticSolve
		select a guess at random
		or enumerate through the value function '''
	if value_fn :
		val = value_fn( poss )
		# remove clues as move options (you cannot change clues)
		clues = poss > 1
		val = val - 100*np.sum(clues, axis=-1)[...,None]
		val = val + np.random.normal(0.0, 0.025, val.shape)
		# val = np.clip(val, -100, 100) # jic
		# val = 1 / (1+np.exp(-val)) # sigmoid, for sampleCategorical
		if False: # DEBUG 
			print("value context:")
			printSudoku("c- ", poss2puzz(poss))
			indx = np.argsort(-val.flatten())
			indx = np.unravel_index(indx, val.shape)
			print("top policy suggestions")
			for i in range(10): 
				print(indx[0][i], indx[1][i], indx[2][i], ":", (indx[0][i]*9+indx[1][i]))
			plt.rcParams['toolbar'] = 'toolbar2'
			plt.imshow(val.reshape((81,9)) )
			plt.title('val')
			plt.show()
		# sel = sampleCategorical( val )
		indx = np.argsort(-val.flatten())
		indx = np.unravel_index(indx, val.shape)
		sel = [indx[0][cntr], indx[1][cntr], indx[2][cntr]]
	else: 
		# pick an undefined cell, set to 1
		sel = random.choice( np.argwhere(poss == 0) )
	guess = np.zeros((9,9,9), dtype=np.int8)
	guess[sel[0], sel[1], sel[2]] = 1
	return guess

def eliminatePoss(poss):
	# apply the rules of Sudoku to eliminate entries in poss.
	poss = np.clip(poss, -1, 1)
	m = poss > 0
	tiles = [(9,1,1), (1,9,1), (1,1,9)]
	for axis in range(3):
		# if a set has an element, eliminate along that axis.
		s = np.sum(m, axis=axis)
		poss = poss*2 - np.tile(np.expand_dims(s, axis=axis), tiles[axis])
		poss = np.clip(poss, -1, 1)
	# do the same for the blocks.
	axis = 1
	poss_box = poss[box_permute[:,0], box_permute[:,1], :].reshape((9,9,9))
	m = poss_box > 0
	s = np.sum(m, axis=axis)
	poss_box = poss_box*2 - np.tile(np.expand_dims(s, axis=axis), tiles[axis])
	poss_box = np.clip(poss_box, -1, 1)
	poss = poss_box[box_unpermute[:,0], box_unpermute[:,1], :].reshape((9,9,9))
	return poss
	
def checkValid(poss): 
	# see if a given poss tensor violates sudoku rules
	m = poss > 0 # positive
	n = poss < 0 # negative
	for axis in range(3): 
		if np.max(np.sum( m, axis=axis)) > 1:
			return False
		if np.max(np.sum( n, axis=axis)) >= 9: # no options
			return False
	# blocks
	poss_box = poss[box_permute[:,0], box_permute[:,1], :].reshape((9,9,9))
	m = poss_box > 0
	n = poss_box < 0
	if np.max(np.sum( m, axis=1)) > 1:
		return False
	if np.max(np.sum( n, axis=1)) >= 9: # no options
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
	return np.sum(poss == 0) > 0
	
def solve(poss, k, record, value_fn, debug=False):
	# input is always valid by construction.
	guesses = np.zeros((9,9,9), dtype=np.int8)
	while canGuess(poss + guesses) and checkValid(poss + guesses) and k > 0:
		# guess = poss2guessSmart(poss + guesses, value_fn)
		guess = poss2guessDumb(poss + guesses, value_fn)
		poss2 = np.array(poss + guesses + guess) 
		# poss2 = eliminatePoss(poss2)
		k = k-1
		if checkValid(poss2): 
			if checkDone(poss2): 
				if debug: 
					print("solved!")
					printSudoku("", poss2puzz(poss2))
				record.append((poss, guesses + guess))
				return True, poss2, k
			else: # not done, recurse
				if debug: 
					print(f"progress: k {k}")
					sel = np.argwhere(guess == 1)
					sel = sel[0]
					printSudoku("", poss2puzz(poss2), curs_pos=sel[:2])
					# printPoss("", poss + guesses + guess)
				if k > 0: 
					valid,ret,k = solve(poss2, k, record, value_fn, debug)
					if valid:
						record.append((poss, guesses + guess))
						# printPoss("", guesses + guess)
						return valid, ret, k
					else:
						# # downstream bad, subtract.
						# don't put on record until we have a positive example
						# record.append((poss, guesses - guess))
						# printPoss("", guesses - guess)
						guesses = guesses - guess
				else: 
					# this guess was valid within the limited roll-out, so add it.
					# terminus of the roll-out is as informed as we can get.. 
					record.append((poss, guesses + guess))
					# printPoss("", guesses + guess)
		else: 
			# one-step bad, subtract.
			guesses = guesses - guess
		guesses = np.clip(guesses, -1, 1)
		
	if debug: 
		print(f"backtracking. canGuess:{canGuess(poss + guesses)} checkValid:{checkValid(poss + guesses)} k:{k} poss+guesses:")
		# printPoss("", poss + guesses)
	if k > 0: 
		return False, poss, k # backtracking
	else:
		return True, poss, k # out of iterations
		
'''
non-backtracking random initial policy:
-- if the board is valid, select a random next move.
-- if the board is invalid,
		select with bias one of the past moves to change.
		Repeat this until the board is valid.
-- accrue statistics on the moves,
		such that you can train a policy for both correct moves and backtracking moves.
-- You cannot change clues. Guesses can be revised.
'''
def stochasticSolve(puzz, n, value_fn, debug=False):
	clues = puzz2poss(puzz) * 2 # clues are 2, guesses are 1.
	clues = np.clip(clues, 0, 2) # replicate training
	iters = 800
	best_i = 0
	guesses_best = None
	for k in range(20): 
		guesses = np.zeros((n, 9,9,9), dtype=np.int8)
		i = 0
		j = 0
		while i < n and j < iters:
			j = j + 1
			poss = np.sum(guesses, axis=0) + clues
			poss_elim = eliminatePoss( np.array(poss) )
			if checkValid(poss) and checkValid(poss_elim):
				if checkDone(poss):
					break
				guess = poss2guessRand(poss, value_fn, 0)
				guesses[i,:,:,:] = guess
				i = i + 1
			else:
				cntr = np.zeros((i), dtype=np.int64)
				while j < iters:
					# try one removal, one addition fixes
					j = j + 1
					s = np.random.randint(0,i)
					fix = guesses[s,:,:,:]*-1 # temp remove past guess
					guess = poss2guessRand(poss + fix, value_fn, cntr[s])
					poss_elim = eliminatePoss( poss + fix + guess ) # this seems like a band-aid.. FIXME.. should learn to detect cycles.
					if checkValid(poss_elim):
						guesses[s,:,:,:] = guess
						break
					else:
						cntr[s] = cntr[s] + 1

			if debug:
				poss = np.sum(guesses, axis=0) + clues
				poss_elim = eliminatePoss( np.array(poss) )
				print(f"i:{i},j:{j}")
				sel = np.where(guess == 1)
				highlight = [sel[0].item(), sel[1].item()]
				printSudoku("",poss2puzz(poss), curs_pos=highlight)
				# printPoss("", poss)
				print(f"checkDone(poss): {checkDone(poss)} checkValid(poss): {checkValid(poss)} checkValid(poss_elim): {checkValid(poss_elim)}")
		if i > best_i: 
			guesses_best = guesses
			best_i = i
	context = np.concatenate((np.expand_dims(clues,0), clues + np.cumsum(guesses_best[:i-1,:,:,:], axis=0)), axis=0)
	return context, guesses_best[:i,:,:,:]


g_puzzles = np.zeros((9,9,9))

def singleSolve(j):
	global g_puzzles
	poss,guess = stochasticSolve(g_puzzles[j], 48, None, False)
	if j%10 == 9:
		print(".", end="", flush=True)
	return (poss,guess)

def parallelSolve(puzzles, N):
	global g_puzzles
	g_puzzles = puzzles
	pool = mp.Pool() #defaults to number of available CPU's
	chunksize = 16
	results = [None for _ in range(N)]
	for ind, res in enumerate(pool.imap_unordered(singleSolve, range(N), chunksize)):
	# for ind in range(N): # debug
	# 	res = singleBacktrack(ind)
		results[ind] = res

	poss_lst = list(map(lambda r: r[0], results))
	guess_lst = list(map(lambda r: r[1], results))
	
	poss_all = np.concatenate(poss_lst)
	guess_all = np.concatenate(guess_lst)

	return poss_all, guess_all
	
	
def encodeSudoku(puzz, top_node=False):
	nodes, _, board_loc = sparse_encoding.puzzleToNodes(puzz, top_node=top_node)
	benc, coo, a2a = sparse_encoding.encodeNodes(nodes)
	return benc, coo, a2a, board_loc
	
	
### parallelize the stochastic solve + value_fn ###
@dataclass
class ValueRequest:
	poss: np.ndarray
	solver_id: int
	request_id: int

@dataclass
class ValueResponse:
	value: np.ndarray
	solver_id: int
	request_id: int
	
@dataclass
class SolveResult:
    puzzle_idx: int
    poss: np.ndarray
    guess: np.ndarray
	
def value_function_worker(value_fn, input_queue, output_queues, active_workers, batch_size=128):
	"""Worker that runs the value function in the main process"""
	pending_requests: List[ValueRequest] = []
	
	while active_workers.value > 0:
		try: # Try to fill up a batch
			while len(pending_requests) < batch_size:
					req = input_queue.get(timeout=0.01)
					if req is None:  # Shutdown signal
						return
					pending_requests.append(req)
		except Empty:
			pass
		
		# Process a batch if we have enough requests or if queue is empty
		if pending_requests and (len(pending_requests) <= batch_size or input_queue.empty()):
			poss_lst = list(map( lambda x: x.poss, pending_requests))
			poss_batch = np.stack(poss_lst)
			value_batch = value_fn( poss_batch, len(poss_lst) )
			
			for i,req in enumerate(pending_requests): 
				resp = ValueResponse(
					value=value_batch[i], 
					solver_id=req.solver_id, 
					request_id=req.request_id )
				output_queues[req.solver_id].put(resp)
			
			pending_requests.clear()
			
def solverWorker(
	puzzles, start_idx, end_idx,
	solver_id: int,
	input_queue: mp.Queue, # to the value fn
	output_queue: mp.Queue, # from the value fn
	result_queue: mp.Queue, # result aggregator
	n_iterations: int,
):
	"""Worker that runs the stochastic solve algorithm"""
	next_request_id = 0
	pending_requests = {}
	
	def value_fn_queue(poss):
		nonlocal next_request_id
		request = ValueRequest(
			poss=poss,
			solver_id=solver_id,
			request_id=next_request_id
		)
		next_request_id += 1
		
		# Send request
		input_queue.put(request)
		response = output_queue.get()  
		return response.value
	
	# Process each puzzle in the assigned chunk
	for idx in range(start_idx, end_idx):
		puzzle = puzzles[idx]
		poss, guess = stochasticSolve(puzzle, n_iterations, value_fn_queue, solver_id==0)
		result_queue.put(SolveResult(puzzle_idx=idx, poss=poss, guess=guess))
	
def parallelSolveVF(
	puzzles: np.ndarray,
	value_fn,
	n_iterations: int = 64,
	n_workers: int = 16,
	batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Solve multiple puzzles in parallel with efficient batching of value function calls.
	"""
	n_puzzles = puzzles.shape[0]
	chunk_size = (n_puzzles + n_workers - 1) // n_workers
	
	# Create queues for communication
	value_fn_input_queue = mp.Queue()
	value_fn_output_queues = {i: mp.Queue() for i in range(n_workers)}
	result_queue = mp.Queue()
	active_workers = mp.Value('i', 1)
	
	# Start solver workers
	solver_processes = []
	for i in range(n_workers):
		start_idx = i * chunk_size
		end_idx = min(start_idx + chunk_size, n_puzzles)
		
		if start_idx >= n_puzzles:
			break
			
		p = mp.Process(
			target=solverWorker,
			args=(
				puzzles, start_idx, end_idx, i,
				value_fn_input_queue, value_fn_output_queues[i],
				result_queue, n_iterations
			)
		)
		p.start()
		solver_processes.append(p)
	
	# Start collecting results in a separate thread 
	results = [None] * n_puzzles
	def result_collector():
		collected = 0
		while collected < n_puzzles:
			try: 
				result = result_queue.get()
				results[result.puzzle_idx] = (result.poss, result.guess)
				collected += 1
				# print(f"+++ got result {collected}")
			except: 
				# Check if all workers are done and we've collected all results
				if collected >= n_puzzles:
					with active_workers.get_lock():
						active_workers.value = 0
		if collected >= n_puzzles:
			with active_workers.get_lock():
				active_workers.value = 0
	
	collector_thread = threading.Thread(target=result_collector)
	collector_thread.start()
	
	# Run value function in main process
	value_function_worker(value_fn, 
							  value_fn_input_queue, 
							  value_fn_output_queues, active_workers, batch_size)
	
	# Clean up
	for p in solver_processes:
		p.join()
	collector_thread.join()
	
	poss_lst = list(map(lambda r: r[0], results))
	guess_lst = list(map(lambda r: r[1], results))
	
	poss_all = np.concatenate(poss_lst)
	guess_all = np.concatenate(guess_lst)

	return poss_all, guess_all
	
def loadRrn(): 
	# use RRN hard to select . 
	csv_file = '../rrn-hard/train.csv'
	base_file = os.path.splitext(csv_file)[0]
	npz_file = f"{base_file}_.npz"
	try:
		file = np.load(npz_file)
		puzzles = file["puzzles"]
	except Exception as error:
		print(error)
		print("Reading %s" % csv_file)
		with open(csv_file) as f:
			puzzles, solutions = [], [] 
			reader = csv.reader(f, delimiter=',')
			for q,a in reader:
				puzzle_digits = list(q)
				puzzles.append(puzzle_digits)
				solution_digits = list(map(int, list(a)))
				solutions.append(solution_digits)
		puzzles = np.stack(puzzles)
		puzzles = puzzles.reshape((puzzles.shape[0],9,9)).astype(np.int8)
		np.savez(npz_file, puzzles=puzzles)
		
	return puzzles

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train sudoku policy model")
	parser.add_argument('-a', action='store_true', help='use AdamW as the optimizer (as opposed to PSGD)')
	parser.add_argument('-c', action='store_true', help="clear, start fresh: don't load model")
	parser.add_argument('-i', type=int, default=0, help='index of computation')
	parser.add_argument('-r', type=int, default=1, help='number of repeats')
	parser.add_argument('--no-train', action='store_true', help="don't train the model.")
	parser.add_argument('--cuda', type=int, default=0, help='index of cuda device')
	cmd_args = parser.parse_args()
	
	print(f"-i: {cmd_args.i} -cuda:{cmd_args.cuda}")
	time.sleep(1)
	
	# increase the number of file descriptors for multiprocessing
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (8*2048, rlimit[1]))

	batch_size = 128
	world_dim = 64
	
	if False: 
		puzz = [
			[0,4,0,0,0,0,0,8,2], 
			[7,0,0,6,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0],
			[0,0,0,0,7,0,0,1,0],
			[0,0,0,0,5,0,6,0,0],
			[0,8,2,0,0,0,0,0,0],
			[3,0,5,0,0,0,7,0,0],
			[6,0,0,1,0,0,0,0,0],
			[0,0,0,8,0,0,0,0,0]] # 17 clues, requires graph coloring. 
		puzz = [
			[0,6,0,0,0,0,1,0,0],
			[0,0,0,3,0,2,0,0,0],
			[0,0,0,0,0,0,0,0,0],
			[0,0,3,0,0,0,0,2,4],
			[8,0,0,0,0,0,0,3,0],
			[0,0,0,0,1,0,0,0,0],
			[0,1,0,0,0,0,7,5,0],
			[2,0,0,9,0,0,0,0,0],
			[0,0,0,4,0,0,6,0,0]]
		puzz = [
			[5,0,7,9,4,0,1,0,0],
			[0,0,0,0,3,0,2,7,6],
			[0,6,0,0,0,8,0,0,5],
			[0,8,0,6,0,4,0,0,0],
			[0,0,5,0,0,0,0,0,3],
			[9,0,0,0,0,0,0,0,2],
			[7,0,8,0,0,9,0,0,0],
			[6,0,0,0,0,0,0,2,9],
			[0,4,0,5,8,1,0,6,0] ]
		puzz = np.array(puzz)
		printSudoku("", puzz)
		start_time = time.time()
		record = []
		_,sol,_ = solve( eliminatePoss(puzz2poss(puzz)), 10000, record, None, True)
		print(f"time: {time.time() - start_time}")
		printSudoku("", poss2puzz(sol))
		
		for i,(poss,guess) in enumerate(record):
			printSudoku("r ",poss2puzz(poss))
			printPoss("", guess)
		# sol = poss2puzz(sol)
		# permute,unpermute = boxPermutation()
		# puzz_perm = sol[permute[:,0], permute[:,1]].reshape((9,9))
		# printSudoku("", puzz_perm)
		# puzz_unperm = puzz_perm[unpermute[:,0],unpermute[:,1]].reshape((9,9))
		# printSudoku("", puzz_unperm)
		exit()

	dat = np.load(f'../satnet/satnet_both_0.65_filled_100000.npz')
	puzzles = dat["puzzles"]
	sudoku = Sudoku(9,60)

	device = torch.device(f'cuda:{cmd_args.cuda}')
	args = {"device": device}
	torch.set_float32_matmul_precision('high')
	torch.backends.cuda.matmul.allow_tf32 = True
	
	memory_dict = gmain.getMemoryDict("../")

	model = Gracoonizer(xfrmr_dim=world_dim, world_dim=world_dim, \
			n_heads=8, n_layers=9, repeat=cmd_args.r, mode=0).to(device)
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
	benc_smol = benc[0].clone().unsqueeze(0)
	# the meat of the ecoding will be replaced by (clipped) poss. 
	
	hcoo = gmain.expandCoordinateVector(coo, a2a, device)
	hcoo = hcoo[0:2] # sparse / set-layers
	hcoo.append('self') # intra-token op
	
	def valueFn(poss, bs): 
		assert(bs <= batch_size)
		poss = torch.from_numpy(poss).float().reshape(bs,81,9)
		poss = poss.to(device)
		benc[:bs,-81:,11:20] = poss
		
		with torch.no_grad(): 
			benc_pred = model.forward(benc, hcoo)
			
		value = benc_pred[:bs,-81:,11:20].cpu().numpy()
		value = np.reshape(value, (bs,9,9,9))
		if False: 
			plt.rcParams['toolbar'] = 'toolbar2'
			fig,axs = plt.subplots(1,3,figsize=(12,6))
			axs[0].imshow(benc[0].cpu().numpy().T)
			axs[1].imshow(benc_pred[0].cpu().numpy().T)
			axs[2].imshow(value[0].reshape(81,9).T)
			plt.show()
		return value
	
	if False:
		n_solved = 0
		record = []
		for i in range(16000, 16001):
			puzz = puzzles[i]
			puzz = np.array([
			[0,4,0,0,0,0,0,8,2], 
			[7,0,0,6,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0],
			[0,0,0,0,7,0,0,1,0],
			[0,0,0,0,5,0,6,0,0],
			[0,8,2,0,0,0,0,0,0],
			[3,0,5,0,0,0,7,0,0],
			[6,0,0,1,0,0,0,0,0],
			[0,0,0,8,0,0,0,0,0]], dtype=np.int8) # 17 clues, requires graph coloring. 
			printSudoku("", puzz)
			poss,guess = stochasticSolve(puzz, 64, valueFn, True)
			sol = poss[-1,:,:,:] + guess[-1,:,:,:]
			printSudoku("", poss2puzz(sol))
			if checkIfValid(sol): 
				n_solved = n_solved + 1
			if i % 10 == 9: 
				print(".", end="", flush=True)
		print(f"n_solved:{n_solved}")
		
		# npz_file = f'satnet_backtrack_0.65.npz'
		# n = len(record)
		# print(f"number of supervised examples: {n}")
		# poss_all = np.zeros((n,9,9,9), dtype=np.int8)
		# guess_all = np.zeros((n,9,9,9), dtype=np.int8)
		# for i,(poss,guess) in enumerate(record):
		# 	# sudoku.printSudoku("r ",poss2puzz(poss))
		# 	# printPoss("", guess)
		# 	poss_all[i] = poss
		# 	guess_all[i] = guess
		# np.savez(npz_file, poss_all=poss_all, guess_all=guess_all)
		exit()
	
	npz_file = f'satnet_backtrack_0.65.npz'
	try:
		file = np.load(npz_file)
		poss_all = file["poss_all"]
		guess_all = file["guess_all"]
		print(f"number of supervised examples: {poss_all.shape[0]}")
	except Exception as error:
		if False: 
			# serial solve, slow.
			rec_poss = []
			rec_guess = []
			for i in range(10000):
				poss, guess = stochasticSolve(puzzles[i], 48, None, False)
				if i % 10 == 9: 
					print(".", end="", flush=True)
				assert(poss.shape[0] == guess.shape[0])
				# for j in range(poss.shape[0]): 
				# 	print("board state:")
				# 	printSudoku("",poss2puzz(poss[j,:,:,:]))
				# 	print("guess:")
				# 	printSudoku("",poss2puzz(guess[j,:,:,:]))
				rec_poss.append(poss)
				rec_guess.append(guess)
			
			poss_all = np.stack(rec_poss)
			guess_all = np.stack(rec_guess)
		else: 
			poss_all, guess_all = parallelSolve(puzzles, 1000)
		
		n = poss_all.shape[0]
		print(f"number of supervised examples: {n}")
		np.savez(npz_file, poss_all=poss_all, guess_all=guess_all)
		
	poss_rrn = []
	guess_rrn = []
	poss_rrn.append(poss_all)
	guess_rrn.append(guess_all)
	try:
		for i in range(4): 
			npz_file = f"rrn_hard_backtrack_{i}.npz"
			file = np.load(npz_file)
			poss_rrn.append(file["poss_all"])
			guess_rrn.append(file["guess_all"])
			print(f"number of supervised examples: {file["poss_all"].shape[0]}")
	except Exception as error:
		print(error)
		puzzles = loadRrn()
		indx = np.random.permutation(puzzles.shape[0])
		sta = cmd_args.i*1024
		poss_rrn, guess_rrn = parallelSolveVF(puzzles[indx[sta:sta+1024],...], valueFn, n_iterations=128, n_workers=batch_size, batch_size=batch_size)
		
		n = poss_rrn.shape[0]
		print(f"number of supervised examples: {n}")
		npz_file = f"rrn_hard_backtrack_{cmd_args.i}.npz"
		np.savez(npz_file, poss_all=poss_rrn, guess_all=guess_rrn)
		
		for i in range(24): 
			printSudoku("", poss2puzz(poss_rrn[i]))
			printSudoku("", poss2puzz(guess_rrn[i]))
			print("")
		exit()
		
	poss_all = np.concatenate(poss_rrn)
	guess_all = np.concatenate(guess_rrn)
	
	DATA_N = poss_all.shape[0]
	VALID_N = DATA_N//10
	indx = np.random.permutation(DATA_N)

	def trainValSplit(y):
		y_train = torch.tensor(y[indx[:-VALID_N]])
		y_valid = torch.tensor(y[indx[-VALID_N:]])
		return y_train, y_valid

	poss_train, poss_valid = trainValSplit(poss_all)
	guess_train, guess_valid = trainValSplit(guess_all)
	assert(guess_train.shape[0] == poss_train.shape[0])

	TRAIN_N = poss_train.shape[0]
	VALID_N = poss_valid.shape[0]

	print(f'loaded {npz_file}; train/test {TRAIN_N} / {VALID_N}')
	
	fd_losslog = open(f'losslog_{utils.getGitCommitHash()}.txt', 'w')
	args['fd_losslog'] = fd_losslog

	if cmd_args.a:
		optimizer_name = "adamw"
	else:
		optimizer_name = "psgd" # adam, adamw, psgd, or sgd
	optimizer = gmain.getOptimizer(optimizer_name, model)

	input_thread = threading.Thread(target=utils.monitorInput, daemon=True)
	input_thread.start()

	bi = TRAIN_N
	avg_duration = 0.0
	
	for uu in range(22000):
		time_start = time.time()
		if bi+batch_size >= TRAIN_N:
			batch_indx = np.random.permutation(TRAIN_N)
			bi = 0
		indx = batch_indx[bi:bi+batch_size]
		bi = bi + batch_size
		
		poss = poss_train[indx, :, :, :].reshape((batch_size,81,9))
		poss = torch.clip(poss, 0, 2)
		guess = guess_train[indx, :, :, :].reshape((batch_size,81,9))
		# model must guess the digit - no longer includes failed attempts
		guess = guess*2 - torch.sum(guess, axis=-1)[...,None]
		mask = guess != 0

		poss = poss.float().to(device)
		guess = guess.float().to(device)
		mask = mask.float().to(device)
		
		benc[:,-81:,11:20] = poss
		benc_new[:,-81:,11:20] = guess + 0.1 # is signed
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
					[torch.sum(1e-4 * \
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
			poss = np.clip(poss, 0, 2)
			guess = guess_valid[indx, :, :, :].reshape((batch_size,81,9))
			guess = guess*2 - torch.sum(guess, axis=-1)[...,None]
			mask = guess != 0

			poss = poss.float().to(device)
			guess = guess.float().to(device)
			mask = mask.float().to(device)
			
			benc[:,-81:,11:20] = poss
			benc_new[:,-81:,11:20] = guess + 0.1 # is signed
			benc_mask[:,-81:,11:20] = mask
			
			benc_pred = model.forward(benc, hcoo)
			
			loss = torch.sum( ((benc_pred - benc_new)*benc_mask)**2 )
			
			lloss = loss.detach().cpu().item()
			print('v',lloss)
			args["fd_losslog"].write(f'{uu}\t{lloss}\t0.0\n')
			args["fd_losslog"].flush()

			uu = uu + 1
