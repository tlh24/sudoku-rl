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

def printSudoku(indent, puzzl_mat, curs_pos=None, highlight_pos=None):
	print(indent, end="  ")
	for j in range(9): 
		print(colored(j, "light_cyan"), end=" ")
	print("")
	for i in range(9):
		print(indent, end="")
		print(colored(i, "light_cyan"), end=" ")
		for j in range(9):
			k = i // 3 + j // 3
			color = "black" if k % 2 == 0 else "red"
			p = int(puzzl_mat[i,j])
			bgcol = None
			if curs_pos is not None: 
				if int(curs_pos[0]) == i and int(curs_pos[1]) == j:
					bgcol = "on_light_yellow"
			if highlight_pos is not None: 
				if int(highlight_pos[0]) == i and int(highlight_pos[1]) == j:
					bgcol = "on_light_blue"
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
	
def poss2guessRand(poss, value_fn, cntr, noise=None):
	''' for use with stochasticSolve
		select a guess at random
		or enumerate through the value function '''
	if value_fn :
		val = value_fn( poss )
		if len(val.shape) == 4: 
			val = val.squeeze()
		# remove clues as move options (you cannot change clues)
		clues = poss > 1
		val = val - 100*np.sum(clues, axis=-1)[...,None]
		# remove existing guesses (you cannot submit same guess twice)
		guesses = poss == 1
		val = val - 100*guesses
		# add a bit of noise, for exploration
		if noise is not None:
			val = val + noise
		# val = np.clip(val, -100, 100) # jic
		# val = 1 / (1+np.exp(-val)) # sigmoid, for sampleCategorical
		if False: # DEBUG 
			print("value context:")
			# printSudoku("c- ", poss2puzz(poss))
			indx = np.argsort(-val.flatten())
			indx = np.unravel_index(indx, val.shape)
			print(f"top policy suggestions; cntr:{cntr}")
			for i in range(10): 
				print(f"r:{indx[0][i]} c:{indx[1][i]} d:{indx[2][i]+1} r*9+c:{(indx[0][i]*9+indx[1][i])}")
			plt.rcParams['toolbar'] = 'toolbar2'
			plt.imshow(np.clip(val.reshape((81,9)).T, -2, 4) )
			plt.title('val')
			plt.colorbar()
			plt.show()
		
		indx = np.argsort(-val.flatten())
		indx = np.unravel_index(indx, val.shape)
		sel = [indx[0][cntr], indx[1][cntr], indx[2][cntr]]
		guess_val = val[indx[0][cntr], indx[1][cntr], indx[2][cntr]]
	else: 
		# pick an undefined cell, set to 1
		sel = random.choice( np.argwhere(poss == 0) )
		guess_val = 1
	guess = np.zeros((9,9,9), dtype=np.int8)
	guess[sel[0], sel[1], sel[2]] = 1
	return guess, guess_val

def eliminatePoss(poss):
	# apply the rules of Sudoku to eliminate entries in poss.
	# aka 'cheating' ;-)
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
	
def pctDone(poss): 
	''' return the fraction done, 0..81 '''
	if not checkValid(poss): 
		return -1 # otherwise the sum would not work! 
	m = poss > 0
	return np.sum( m )  
	
def canGuess(poss): 
	return np.sum(poss == 0) > 0
		

'''
experimentation-based stochastic solve.  
as soon as you hit an invalid board, 
select one past guess to experiment on.  
for that guess, select n alternatives, and run the deterministic 
solver until it hits an invalid board. 
if all the alternatives are the same, pick another guess to experiment on.
'''
def experimentSolve(puzz, n, value_fn, debug=False):
	clues = puzz2poss(puzz) * 2 # clues are 2, guesses are 1.
	clues = np.clip(clues, 0, 2) # needed, o/w get -2
	clues = clues.astype(np.int8) # o/w is int64
	guesses = np.zeros((81,9,9,9), dtype=np.int8)
	advantage = None
	clues_fill = pctDone( clues )
	i = 0
	while i < 81:
		poss = np.sum(guesses, axis=0) + clues
		if checkValid(poss):
			if checkDone(poss):
				break
			guess,_ = poss2guessRand(poss, value_fn, 0)
			guesses[i,:,:,:] = guess
			i += 1
		else:
			exp_guess_list = []
			context_list = []
			ss = np.random.permutation(i) 
			
			for si in range(min(i, 16)): # iterate over different guesses
				# s = ss[si]
				s = si # FIXME
				different = False
				std = 0.1
				
				exp_guess = np.zeros((n,9,9,9), dtype=np.int8)
				guesses_test = np.zeros((81,9,9,9), dtype=np.int8)
				advantage = np.zeros((n,), dtype=int)
				
				while not different: 
					# test n possible replacements @ s
					noise = np.random.normal(0.0, std, (9,9,9) )
					for k in range(n): 
						guesses_test[:,:,:,:] = 0 # erase all
						guesses_test[0:s,:,:,:] = guesses[0:s,:,:,:] 
						poss = np.sum(guesses_test, axis=0) + clues
						guess,_ = poss2guessRand(poss, value_fn, k, noise)
						guesses_test[s,...] = guess
						exp_guess[k,...] = guess
						# do deterministic roll-outs from this replacement
						#  (which involves one redo, but ok)
						poss = np.sum(guesses_test, axis=0) + clues
						while checkValid(poss): 
							advantage[k] += 1
							if checkDone(poss):
								break
							assert(advantage[k] + s + clues_fill <= 81)
							guess,_ = poss2guessRand(poss, value_fn, 0)
							guesses_test[s+advantage[k],...] = guess
							poss = np.sum(guesses_test, axis=0) + clues
					different = np.var(advantage) > 0
					std *= 1.1
					
				if different: 
					if s == 0: 
						context = clues
					if s > 0: 
						context = np.sum(guesses_test[:s,...], axis=0) + clues
					advantage_f = advantage - np.max(advantage) 
					# advantage = np.exp(advantage / min(np.std(advantage), 4.0))
					advantage_f = np.exp(advantage_f * 3) # hard label the max! 
					if debug:
						if s > 0: 
							cp_r,cp_c,_ = np.where(guesses_test[s-1,:,:,:] == 1)
							curs_pos = [cp_r.item(), cp_c.item()]
						else: curs_pos = None
						printSudoku("c- ",poss2puzz(context), curs_pos)
						print("clues:", clues_fill, "s:", s, advantage + clues_fill + s)
						for k in range(n): 
							row,col,dig = np.where(exp_guess[k,:,:,:] == 1)
							row = row.item()
							col = col.item()
							dig = dig.item()
							af = int(np.round(advantage_f[k]))
							print("a-", af, advantage[k] + clues_fill + s, "@", row,col,dig+1)
					
					# convert advantage to fixed-point
					advantage = np.clip(advantage, 1/127, 1)*127
					exp_guess = np.sum(exp_guess * advantage[:,np.newaxis,np.newaxis,np.newaxis], axis=0)
					exp_guess = np.clip(exp_guess, -127, 127) # prevent wrap
					
					context_list.append(context.astype(np.int8))
					exp_guess_list.append(exp_guess.astype(np.int8))
				
			break # end for; break out of while i < 81: got the data

	# if we finished the puzzle, keep around as positive training: 
	if advantage is None: 
		if debug: print("solved.")
		context = np.concatenate(
			(np.expand_dims(clues,0), # starting board
				clues +
					np.cumsum( # next series of boards
						np.clip( guesses[:i-1,:,:,:], 0, 1), # don't add last guess
					axis=0)),
			axis=0) # cumsum outputs int64.  won't wrap on conversion to int8.
		# fixed point -- after the cumsum! 
		guesses[:i,:,:,:] = guesses[:i,:,:,:]*127
		return context.astype(np.int8), guesses[:i,:,:,:].astype(np.int8)
	else: 
		context_out = np.stack(context_list)
		exp_guess_out = np.stack(exp_guess_list)
		return context_out, exp_guess_out


g_puzzles = np.zeros((9,9,9))

def singleSolve(j):
	global g_puzzles
	poss,guess = experimentSolve(g_puzzles[j], 20, None, False)
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
	
	while active_workers.value > 0: # no write / no need for a lock.
		try: # Try to fill up a batch
			while len(pending_requests) < batch_size:
				req = input_queue.get(timeout=0.003)
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
	active_workers: mp.Value
):
	"""Worker that runs the experiment solve algorithm"""
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
		assert(response.request_id == request.request_id)
		assert(response.solver_id == solver_id)
		return response.value
	
	# Process each puzzle in the assigned chunk
	for idx in range(start_idx, end_idx):
		puzzle = puzzles[idx]
		poss, guess = experimentSolve(puzzle, n_iterations, value_fn_queue, solver_id==0)
		result_queue.put(SolveResult(puzzle_idx=idx, poss=poss, guess=guess))
		if solver_id == 0: 
			print(f"{idx}/{end_idx} {math.floor(100*(idx-start_idx)/(end_idx-start_idx))}" )

	# done, so decrement active_workers
	with active_workers.get_lock():
		active_workers.value -= 1
		print(f"solver_worker done {active_workers.value}")
	return None
	
def parallelSolveVF(
	puzzles: np.ndarray,
	value_fn,
	n_iterations: int = 20,
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
	active_workers = mp.Value('i', n_workers)
	
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
				result_queue, n_iterations, active_workers
			)
		)
		p.start()
		solver_processes.append(p)
	
	# Start collecting results in a separate thread 
	results = []
	def result_collector():
		done = False
		while active_workers.value > 0:
			try: 
				result = result_queue.get()
				results.append( (result.poss, result.guess) )
				collected += 1
			except: 
				time.sleep(0.001)
		print("result_collector done")
		return None
	
	collector_thread = threading.Thread(target=result_collector)
	collector_thread.start()
	
	# Run value function in main process
	value_function_worker(value_fn, 
							  value_fn_input_queue, 
							  value_fn_output_queues, active_workers, batch_size//2)
	
	# Clean up
	collector_thread.join()
	print("collector_thread joined")
	for p in solver_processes:
		p.join(timeout=1)
		print("joined")
		if p.is_alive():
			print("Process is still running. Terminating it now.")
			p.terminate()  # Forcefully terminate the process
			p.join()       # Join again to clean up resources after termination
	
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
	parser.add_argument('-r', type=int, default=2, help='number of repeats')
	parser.add_argument('--no-train', action='store_true', help="don't train the model.")
	parser.add_argument('--cuda', type=int, default=0, help='index of cuda device')
	parser.add_argument('-b', type=int, default=128, help='batch size')
	parser.add_argument('--puzz', type=int, default=12, help='number of puzzles to solve, in units of 1024')
	cmd_args = parser.parse_args()
	
	print(f"-i: {cmd_args.i} -cuda:{cmd_args.cuda}")
	time.sleep(1)
	
	# increase the number of file descriptors for multiprocessing
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (8*2048, rlimit[1]))

	batch_size = cmd_args.b
	world_dim = 64

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
	benc, coo, a2a, board_loc = encodeSudoku(np.zeros((9,9)))
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
	
	def valueFn(poss, bs=1): 
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
	
	if False: # development & debugging
		dat = np.load(f'../satnet/satnet_both_0.65_filled_100000.npz')
		puzzles = dat["puzzles"]
		n_solved = 0
		record = []
		# parallelSolveVF(puzzles[:128,...], valueFn, n_iterations=20, n_workers=16, batch_size=16)
		# exit()
		for i in range(0, 25):
			puzzles = [ [
			[0,4,0,0,0,0,0,8,2],
			[7,0,0,6,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0],
			[0,0,0,0,7,0,0,1,0],
			[0,0,0,0,5,0,6,0,0],
			[0,8,2,0,0,0,0,0,0],
			[3,0,5,0,0,0,7,0,0],
			[6,0,0,1,0,0,0,0,0],
			[0,0,0,8,0,0,0,0,0]], # 17 clues, requires graph coloring.
			[
			[0,6,0,0,0,0,1,0,0],
			[0,0,0,3,0,2,0,0,0],
			[0,0,0,0,0,0,0,0,0],
			[0,0,3,0,0,0,0,2,4],
			[8,0,0,0,0,0,0,3,0],
			[0,0,0,0,1,0,0,0,0],
			[0,1,0,0,0,0,7,5,0],
			[2,0,0,9,0,0,0,0,0],
			[0,0,0,4,0,0,6,0,0]],
			[
			[5,0,7,9,4,0,1,0,0],
			[0,0,0,0,3,0,2,7,6],
			[0,6,0,0,0,8,0,0,5],
			[0,8,0,6,0,4,0,0,0],
			[0,0,5,0,0,0,0,0,3],
			[9,0,0,0,0,0,0,0,2],
			[7,0,8,0,0,9,0,0,0],
			[6,0,0,0,0,0,0,2,9],
			[0,4,0,5,8,1,0,6,0] ] ]
			puzzles = np.array(puzzles)
			# puzzles = loadRrn()
			puzz = puzzles[i%puzzles.shape[0] ]
			printSudoku("", puzz)
			poss,guess = experimentSolve(puzz, 20, valueFn, True)
		exit()
		
	poss_rrn = []
	guess_rrn = []
	try:
		for i in range(4): 
			npz_file = f"rrn_hard_backtrack_{i}.npz"
			file = np.load(npz_file)
			poss_rrn.append(file["poss_all"])
			guess_rrn.append(file["guess_all"])
			n_data = file["poss_all"].shape[0]
			print(f"number of supervised examples: {n_data}")
	except Exception as error:
		print(error)
		# puzzles = loadRrn()
		dat = np.load(f'../satnet/satnet_both_0.65_filled_100000.npz')
		puzzles = dat["puzzles"]
		indx = np.random.permutation(puzzles.shape[0])
		incr = 1024*cmd_args.puzz
		sta = cmd_args.i*incr
		puzzles_permute = np.array(puzzles[indx,...])
		poss_rrn, guess_rrn = parallelSolveVF(puzzles_permute[sta:sta+incr,...], valueFn, n_iterations=20, n_workers=batch_size, batch_size=batch_size)
		
		n = poss_rrn.shape[0]
		print(f"number of supervised examples: {n}")
		npz_file = f"rrn_hard_backtrack_{cmd_args.i}.npz"
		np.savez(npz_file, poss_all=poss_rrn, guess_all=guess_rrn)
		
		# if cmd_args.c == 0 and cmd_args.i == 0:
		# 	for i in range(100):
		# 		printSudoku("", poss2puzz(poss_rrn[i]))
		# 		printPoss("", guess_rrn[i])
		# 		print("")
		exit()
		
	poss_all = np.concatenate(poss_rrn)
	guess_all = np.concatenate(guess_rrn)

	# if there is one guess, need to set other digits in that cell to 0. 
	# (these are from successful boards)
	one_guess = np.sum(np.abs(guess_all), axis=(1,2,3))
	y = (np.sum(guess_all, axis=3) / 127.0) \
			* (one_guess == 127)[:,np.newaxis,np.newaxis]
	y = np.tile(y[:,:,:,np.newaxis], (1,1,1,9)).astype(np.int8)
	guess_all = guess_all - y # other options in cell are -1/127
	
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
	# optimizer.lr_params = 0.001
	# optimizer.lr_preconditioner=0.002
	# optimizer.momentum=0.8
	
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
		# guess = guess*2 - torch.sum(guess, axis=-1)[...,None]
		mask = guess != 0

		poss = poss.float().to(device)
		guess = guess.float().to(device)
		guess = guess / 127 # fixed to floating point
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
			# guess = guess*2 - torch.sum(guess, axis=-1)[...,None]
			mask = guess != 0

			poss = poss.float().to(device)
			guess = guess.float().to(device)
			guess = guess / 127 # fixed to floating point
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
