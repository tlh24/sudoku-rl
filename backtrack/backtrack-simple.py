import numpy as np
import time
from termcolor import colored

def isValid(board, row, col, num):
	# Check if the number is not repeated in the current row
	if num in board[row, :]:
		return False
	# Check if the number is not repeated in the current column
	if num in board[:, col]:
		return False
	# Check if the number is not repeated in the 3x3 subgrid
	start_row, start_col = 3 * (row // 3), 3 * (col // 3)
	if num in board[start_row:start_row+3, start_col:start_col+3]:
		return False

	return True

def findEmpty(board):
	# Find an empty cell (with a value of 0)
	for i in range(9):
		for j in range(9):
			if board[i, j] == 0:
				return i, j
	return None

g_evals = 0
g_backtrack = 0

def sudokuSolver(board):
	global g_evals, g_backtrack
	# ignore the number of evals to find an empty spot.
	# note find_empty_location is *ordered*
	empty_loc = findEmpty(board)

	# If no empty cells remain, the puzzle is solved
	if not empty_loc:
		return True
	row, col = empty_loc

	# Try placing numbers 1 through 9 in the empty cell
	for num in range(1, 10):
		g_evals = g_evals + 1
		if isValid(board, row, col, num):
			board[row, col] = num

			# Recursively attempt to solve the rest of the board
			done = sudokuSolver(board)
			if done:
					return True

			# If placing num doesn't lead to a solution, backtrack
			board[row, col] = 0
			g_backtrack = g_backtrack + 1
			# if g_backtrack % 1000 == 999:
			# 	print(".", end="", flush=True)

	# Trigger backtracking
	return False

board_strs = []
board_descs = []

board_descs.append('''
	 This puzzle is both very hard -
	 requires many applications of inference chains -
	 and is degenerate, with 26 solutions.
	 It can be solved relatively quickly with backtracking.
	 ''')
board_strs.append( "000060300" +\
		"000500020" +\
		"106000070" +\
		"370002000" +\
		"400003000" +\
		"000958000" +\
		"090000000" +\
		"080020005" +\
		"003000980" )

board_descs.append('''
	 This 17-clue puzzle is hard,
	 but does not require any advanced strategies or graph coloring.
	 just hidden singles, doubles, and triples.
	 ''')
board_strs.append( "107200000" +\
		"000050400" +\
		"000100000" +\
		"450000600" +\
		"000700080" +\
		"030000000" +\
		"600034000" +\
		"000000071" +\
		"000000000" )

board_descs.append('''
	 Another 17 clues.
	 this puzzle is relatively easy,
	 and only requies the hidden singles strategy.
	 nonetheless, it requires very extensive backtracking.
	 ''')
board_strs.append( "500070600" +\
		"000010000" +\
		"000000800" +\
		"005009002" +\
		"400800000" +\
		"000000010" +\
		"010200000" +\
		"000300500" +\
		"700000040" )

board_descs.append('''
	 17-clue puzzle no 3.
	 Requires hidden singles, doubles, triples,
	 but no graph coloring or other strategies.
	 Very slow to solve due to the many zeros on the first row,
	 hence much backtracking with the ordered algorithm.
	 ''')
board_strs.append( "060000100" +\
		"000302000" +\
		"000000000" +\
		"003000024" +\
		"800000030" +\
		"000010000" +\
		"010000750" +\
		"200900000" +\
		"000400600" )

for board_str, board_desc in zip(board_strs, board_descs):
	board_int = [int(c) for c in board_str]
	board = np.array(board_int).reshape(9,9)
	clues = np.sum(board > 0)
	print(f"puzzle ({clues} clues):")
	print(board_desc)
	print(board_str)
	print(board)

	g_evals = 0
	g_backtrack = 0
	time_start = time.time()
	done = sudokuSolver(board)
	time_end = time.time()
	if done:
		print("Solution:")
		print(board)
		print("number of evals: ", end="")
		print(colored(f"{g_evals}", attrs=["bold"]))
		print("backtrack: ", end="")
		print(colored(f"{g_backtrack}", attrs=["bold"]))
		print("time to solve: ", end="")
		print(colored(f"{time_end - time_start}", attrs=["bold"]))
		print("")
	else:
		print("No solution exists.")
