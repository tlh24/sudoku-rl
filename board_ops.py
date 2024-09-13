import numpy as np
import torch
from termcolor import colored
from multiprocessing import Pool
from type_file import Action, Axes, getActionName
from sudoku_gen import Sudoku
from constants import SuN,SuK,g_dtype,token_cnt
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


def encodeBoard(sudoku, puzzl_mat, guess_mat, curs_pos, action, action_val, many_reward=False):
	'''
	Encodes the current board state and encodes the given action,
		runs the action, and then encodes the new board state.

	The board and action nodes have the same encoding- contains one hot of node type and node value

	Returns:
	board encoding: Shape (#board nodes x world_dim)
	action encoding: Shape (#action nodes x world_dim)
	new board encoding: Shape (#newboard nodes x world_dim)
	'''
	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action, action_val, 0.0, many_reward)
	benc,coo,a2a = sparse_encoding.encodeNodes(nodes)

	reward = runAction(sudoku, puzzl_mat, guess_mat, curs_pos, action, action_val)

	nodes, reward_loc,_ = sparse_encoding.sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action, action_val, reward, many_reward) # action_val doesn't matter
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

def enumerateActions(arg):
	i_p, puzzl = arg
	action_types,action_values = enumerateActionList()
	n_actions = len(action_types) # 13
	n_curspos = 3
	n_masks = 3
	sudoku = Sudoku(SuN, SuK)
	sn = n_actions * n_masks * n_curspos
	i = 0
	# print(f"new tensor size {sn} {token_cnt} 32")
	# !!! torch does not work with multiprocessing !!!
	orig_boards_l = np.zeros((sn, token_cnt, 32), dtype=np.float16)
	new_boards_l = np.zeros((sn, token_cnt, 32), dtype=np.float16)
	rewards_l = np.zeros((sn,), dtype=np.float16)
	for i_m in range(n_masks):
		# move half the clues to guesses (on average)
		# to force generalization over both!
		mask = np.random.randint(0,2, (SuN,SuN)) == 1
		# mask = np.zeros((SuN,SuN))
		guess_mat = puzzl * mask
		puzzl_mat = puzzl * (1-mask)
		for i_c in range(n_curspos):
			curs_pos = np.random.randint(0, SuN, (2,))
			# for half the boards, select only open positions.
			if (i_c + i_m*3)%2 == 1:
				while puzzl[curs_pos[0], curs_pos[1]] > 0:
					curs_pos = np.random.randint(0, SuN, (2,), dtype=int)
			for i_a in range(n_actions):
				mask = np.random.randint(0,2, (SuN,SuN)) == 1
				# mask = np.zeros((SuN,SuN))
				guess_mat = puzzl * mask
				puzzl_mat = puzzl * (1-mask)
				at,av = action_types[i_a], action_values[i_a]

				benc,newbenc,coo,a2a,reward,reward_loc = \
					encodeBoard(sudoku, puzzl_mat, guess_mat, curs_pos, at, av )
				orig_boards_l[i] = benc.astype(np.float16)
				new_boards_l[i] = newbenc.astype(np.float16)
				rewards_l[i] = reward
				if i % 100 == 99:
					print(".", end = "", flush=True)
				i = i + 1
	return (orig_boards_l, new_boards_l, rewards_l)


def enumerateBoards(puzzles):
	# changing the strategy: for each board, do all possible actions.
	# this serves as a stronger set of constraints than random enumeration.
	action_types,action_values = enumerateActionList()
	n_actions = len(action_types) # 13
	n_curspos = 3
	n_masks = 3
	n_puzzles = 2048 # 1024, 1280 1536 2048
	n = n_actions * n_masks * n_curspos * n_puzzles
	
	try:
		orig_board_enc = torch.load(f'orig_board_enc_{n}.pt',weights_only=True)
		new_board_enc = torch.load(f'new_board_enc_{n}.pt',weights_only=True)
		rewards_enc = torch.load(f'rewards_enc_{n}.pt',weights_only=True)
		n = 1
	except Exception as error:
		print(colored(f"could not load precomputed data {error}", "red"))
		print("generating random board, action, board', reward")

	if n > 1:
		orig_board_enc = np.zeros((n, token_cnt, 32), dtype=np.float16)
		new_board_enc = np.zeros((n, token_cnt, 32), dtype=np.float16)
		rewards_enc = np.zeros((n,), dtype=np.float16)

		sn = n_actions * n_masks * n_curspos
		args = []
		for i in range(n_puzzles):
			args.append((i, puzzles[i].numpy()))
		chunksize = 1
		pool = Pool() #defaults to number of available CPU's
		for ind, res in enumerate(pool.imap_unordered(enumerateActions, args, chunksize)):
		# for ind in range(n_puzzles):
			# res = enumerateActions(args[ind])
			orig_boards_l, new_boards_l, rewards_l = res
			orig_board_enc[sn*ind:sn*(ind+1),:,:] = orig_boards_l
			new_board_enc[sn*ind:sn*(ind+1),:,:] = new_boards_l
			rewards_enc[sn*ind:sn*(ind+1)] = rewards_l

		orig_board_enc = torch.from_numpy(orig_board_enc)
		new_board_enc = torch.from_numpy(new_board_enc)
		rewards_enc = torch.from_numpy(rewards_enc)

		print("saving the generated boards")
		torch.save(orig_board_enc, f'orig_board_enc_{n}.pt')
		torch.save(new_board_enc, f'new_board_enc_{n}.pt')
		torch.save(rewards_enc, f'rewards_enc_{n}.pt')


	# for i_p in range(n_puzzles):
	# 	puzzl = puzzles[i_p, :, :].numpy()
	# 	for i_m in range(3):
	# 		# move half the clues to guesses (on average)
	# 		# to force generalization over both!
	# 		mask = np.random.randint(0,2, (SuN,SuN)) == 1
	# 		# mask = np.zeros((SuN,SuN))
	# 		guess_mat = puzzl * mask
	# 		puzzl_mat = puzzl * (1-mask)
	# 		for i_c in range(3):
	# 			curs_pos = torch.randint(SuN, (2,), dtype=int)
	# 			# for half the boards, select only open positions.
	# 			if (i_c + i_m*3)%2 == 1:
	# 				while puzzl[curs_pos[0], curs_pos[1]] > 0:
	# 					curs_pos = torch.randint(SuN, (2,), dtype=int)
	# 			for i_a in range(n_actions):
	# 				mask = np.random.randint(0,2, (SuN,SuN)) == 1
	# 				# mask = np.zeros((SuN,SuN))
	# 				guess_mat = puzzl * mask
	# 				puzzl_mat = puzzl * (1-mask)
	# 				at,av = action_types[i_a], action_values[i_a]
 #
	# 				benc,newbenc,coo,a2a,reward,reward_loc = \
	# 					encodeBoard(sudoku, puzzl_mat, guess_mat, curs_pos, at, av )
	# 				orig_boards.append(benc)
	# 				new_boards.append(newbenc)
	# 				rewards[i] = reward
	# 				if i % 1000 == 999:
	# 					print(".", end = "", flush=True)
	# 				i = i + 1

	# if n_puzzles > 1:

	return orig_board_enc, new_board_enc, rewards_enc
