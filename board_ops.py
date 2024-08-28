import numpy as np
import torch
from termcolor import colored
from type_file import Action, Axes, getActionName
from sudoku_gen import Sudoku
from constants import SuN,SuK,g_dtype
import sparse_encoding
import pdb

'''
	Sudoku board-related functions
'''


def runAction(sudoku, puzzl_mat, guess_mat, curs_pos, action:int, action_val:int):
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
	# clip (rather than wrap) cursor position
	for i in range(2):
		if curs_pos[i] < 0:
			reward = -0.5
			curs_pos[i] = 0
		if curs_pos[i] >= SuN:
			reward = -0.5
			curs_pos[i] = SuN - 1

	# if we're on a open cell, but no moves are possible,
	# negative reward!
	clue = puzzl_mat[curs_pos[0], curs_pos[1]]
	curr = guess_mat[curs_pos[0], curs_pos[1]]
	sudoku.setMat(puzzl_mat + guess_mat) # so that checkIfSafe works properly.
	if clue == 0 and curr == 0:
		if not sudoku.checkOpen(curs_pos[0], curs_pos[1]):
			print(" ")
			sudoku.printSudoku("", puzzl_mat, guess_mat, curs_pos)
			pdb.set_trace() # should not happen!
			reward = -1

	if action == Action.SET_GUESS.value:
		if clue == 0 and curr == 0 and sudoku.checkIfSafe(curs_pos[0], curs_pos[1], action_val):
			# updateNotes(curs_pos, action_val, notes)
			reward = 1
			guess_mat[curs_pos[0], curs_pos[1]] = action_val
		else:
			reward = -1
	if action == Action.UNSET_GUESS.value:
		if curr != 0:
			guess_mat[curs_pos[0], curs_pos[1]] = 0
			reward = -1 # must exactly cancel, o/w best strategy is to simply set/unset guess repeatedly.
		else:
			reward = -1.25

	if False:
		print(f'runAction @ {curs_pos[0]},{curs_pos[1]}: {action}:{action_val}')

	return reward

def oneHotEncodeBoard(sudoku, curs_pos, action: int, action_val: int, enc_dim: int = 20):
	'''
	** Deprecated **
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

	reward = runAction(sudoku, None, curs_pos, action, action_val)

	new_curs_enc = curs_enc + action_enc

	return curs_enc, action_enc, new_curs_enc, mask, reward


def encodeBoard(sudoku, puzzl_mat, guess_mat, curs_pos, action, action_val):
	'''
	Encodes the current board state and encodes the given action,
		runs the action, and then encodes the new board state.

	The board and action nodes have the same encoding- contains one hot of node type and node value

	Returns:
	board encoding: Shape (#board nodes x world_dim)
	action encoding: Shape (#action nodes x world_dim)
	new board encoding: Shape (#newboard nodes x world_dim)
	'''
	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action, action_val, 0.0)
	benc,coo,a2a = sparse_encoding.encodeNodes(nodes)

	reward = runAction(sudoku, puzzl_mat, guess_mat, curs_pos, action, action_val)

	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action, action_val, reward) # action_val doesn't matter
	newbenc,coo,a2a = sparse_encoding.encodeNodes(nodes)

	return benc, newbenc, coo, a2a, reward, reward_loc

def encode1DBoard():
	# simple 1-D version of sudoku.
	puzzle = np.arange(1, 10)
	mask = np.random.randint(0,3,9)
	puzzle = puzzle * (mask > 0)
	curs_pos = np.random.randint(0,9)
	action = 4
	action_val = np.random.randint(0,9)
	guess_mat = np.zeros((9,))

	nodes, reward_loc,_ = sparse_encoding.sudoku1DToNodes(puzzle, guess_mat, curs_pos, action, action_val, 0.0)
	benc,coo,a2a = sparse_encoding.encodeNodes(nodes)

	# run the action.
	reward = -1
	if puzzle[action_val-1] == 0 and puzzle[curs_pos] == 0:
		guess_mat[curs_pos] = action_val
		reward = 1

	nodes, reward_loc,_ = sparse_encoding.sudoku1DToNodes(puzzle, guess_mat, curs_pos, action, action_val, reward) # action_val doesn't matter
	newbenc,coo,a2a = sparse_encoding.encodeNodes(nodes)

	return benc, newbenc, coo, a2a, reward, reward_loc


def enumerateActionList():
	action_types = []
	action_values = []
	# directions
	for at in [0,1,2,3]:
		action_types.append(at)
		action_values.append(0)
	at = Action.SET_GUESS.value
	for av in range(SuN):
		action_types.append(at)
		action_values.append(av+1)
	# # unset guess action
	# action_types.append( Action.UNSET_GUESS.value )
	# action_values.append( 0 )
	# the inverse model has a hard time understanding UNSET_GUESS
	# it's encoded in the same was as guess, only with zero as an arg.

	return action_types,action_values

def sampleActionList(n:int):
	# this is slow but whatever, only needs to run once
	action_types = []
	action_values = []
	possible_actions = [ 0,1,2,3,4,4,4,4,4,5,5 ] # FIXME
	for i in range(n):
		action = possible_actions[np.random.randint(len(possible_actions))]
		actval = 0
		if action == Action.SET_GUESS.value:
			actval = np.random.randint(1,10)
		action_types.append(action)
		action_values.append(actval)

	return action_types,action_values


def enumerateBoards(puzzles):
	# changing the strategy: for each board, do all possible actions.
	# this serves as a stronger set of constraints than random enumeration.
	action_types,action_values = enumerateActionList()
	n_actions = len(action_types) # 13
	n_curspos = 3
	n_masks = 3
	n_puzzles = 1280 # 1280 or 1024
	n = n_actions * n_masks * n_curspos * n_puzzles
	
	try:
		orig_board_enc = torch.load(f'orig_board_enc_{n}.pt',weights_only=True)
		new_board_enc = torch.load(f'new_board_enc_{n}.pt',weights_only=True)
		rewards_enc = torch.load(f'rewards_enc_{n}.pt',weights_only=True)
		# need to get the coo, a2a, etc variables - so run one encoding.
		n_puzzles = 1
	except Exception as error:
		print(colored(f"could not load precomputed data {error}", "red"))
		print("generating random board, action, board', reward")

	sudoku = Sudoku(SuN, SuK)
	orig_boards = []
	new_boards = []
	rewards = torch.zeros(n, dtype=g_dtype)
	i = 0
	for i_p in range(n_puzzles): 
		puzzl = puzzles[i_p, :, :].numpy()
		for i_m in range(3): 
			# move half the clues to guesses (on average)
			# to force generalization over both!
			mask = np.random.randint(0,2, (SuN,SuN)) == 1
			# mask = np.zeros((SuN,SuN))
			guess_mat = puzzl * mask
			puzzl_mat = puzzl * (1-mask)
			for i_c in range(3): 
				curs_pos = torch.randint(SuN, (2,), dtype=int)
				# for half the boards, select only open positions.
				if (i_c + i_m*3)%2 == 1: 
					while puzzl[curs_pos[0], curs_pos[1]] > 0:
						curs_pos = torch.randint(SuN, (2,), dtype=int)
				for i_a in range(n_actions):
					mask = np.random.randint(0,2, (SuN,SuN)) == 1
					# mask = np.zeros((SuN,SuN))
					guess_mat = puzzl * mask
					puzzl_mat = puzzl * (1-mask)
					at,av = action_types[i_a], action_values[i_a]

					benc,newbenc,coo,a2a,reward,reward_loc = \
						encodeBoard(sudoku, puzzl_mat, guess_mat, curs_pos, at, av )
					orig_boards.append(benc)
					new_boards.append(newbenc)
					rewards[i] = reward
					if i % 1000 == 999:
						print(".", end = "", flush=True)
					i = i + 1

	if n_puzzles > 1: 
		orig_board_enc = torch.stack(orig_boards)
		new_board_enc = torch.stack(new_boards)
		rewards_enc = rewards
		torch.save(orig_board_enc, f'orig_board_enc_{n}.pt')
		torch.save(new_board_enc, f'new_board_enc_{n}.pt')
		torch.save(rewards_enc, f'rewards_enc_{n}.pt')

	return orig_board_enc, new_board_enc, coo, a2a, rewards_enc, reward_loc
