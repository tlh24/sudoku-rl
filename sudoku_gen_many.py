'''
Generates N sudoku maps with some positions blank and some filled. 
Saves the tensor of maps as a torch file named puzzles_{N}.pt
'''
import math
import torch
import numpy as np
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product
import random, copy
import os 

def makePuzzle(j): # argument is ignored.
	S = 9
	if S == 9:
		k = np.random.randint(20) + 25 # how many positions to blank.
	if S == 4:
		k = np.random.randint(4) + 5
	sudoku = Sudoku(S, k)
	sudoku.fillValues()
	return torch.tensor(sudoku.mat)

def generatePuzzles(N=500000,S=9):
	'''
	Generate puzzles following Tim's puzzle generation code 
	'''
	# the naive sudoku generator is slow. 
	# parallelize and save the results for fast loading later.
	x = torch.zeros(N, S, S)

	pool = Pool() #defaults to number of available CPU's
	chunksize = 1 # some puzzles require a lot of backtracking to fill, so keep small
	for ind, res in enumerate(pool.imap_unordered(makePuzzle, range(N), chunksize)):
		x[ind, :, :] = res
    
	torch.save(x, f'puzzles_{S}_{N}.pt')

# dumb global -- need it for multiprocessing.
percent_filled = 0.75

def generateSATNetPuzzles(_n):
	'''
	Generates 9x9 sudoku boards and their solutions. Adaped from https://github.com/Kyubyong/sudoku
	'''
	def construct_puzzle_solution():
		# Loop until we're able to fill all 81 cells with numbers, while
		# satisfying the constraints above.
		while True:
			try:
				puzzle  = [[0]*9 for i in range(9)] # start with blank puzzle
				rows    = [set(range(1,10)) for i in range(9)] # set of available
				columns = [set(range(1,10)) for i in range(9)] #   numbers for each
				squares = [set(range(1,10)) for i in range(9)] #   row, column and square
				for i in range(9):
					for j in range(9):
						# pick a number for cell (i,j) from the set of remaining available numbers
						choices = rows[i].intersection(columns[j]).intersection(squares[(i//3)*3 + j//3])
						choice  = random.choice(list(choices))
			
						puzzle[i][j] = choice
			
						rows[i].discard(choice)
						columns[j].discard(choice)
						squares[(i//3)*3 + j//3].discard(choice)

				# success! every cell is filled.
				return puzzle
				
			except IndexError:
				# if there is an IndexError, we have worked ourselves in a corner (we just start over)
				pass


	def run(num_given_cells, iter=10):
		'''
		Attempts to create a puzzle with num_given_cells, but if it can't returns puzzle
			as close to that and larger in set of iter puzzles generated
		'''
		all_results = {}
	#     print "Constructing a sudoku puzzle."
	#     print "* creating the solution..."
		a_puzzle_solution = construct_puzzle_solution()
		
	#     print "* constructing a puzzle..."
		for i in range(iter):
			puzzle = copy.deepcopy(a_puzzle_solution)
			(result, number_of_cells) = pluck(puzzle, num_given_cells)
			all_results.setdefault(number_of_cells, []).append(result)
			if number_of_cells <= num_given_cells: break
	
		return all_results, a_puzzle_solution

	def best(set_of_puzzles):
		# Could run some evaluation function here. For now just pick
		# the one with the fewest "givens".
		return set_of_puzzles[min(set_of_puzzles.keys())][0]

	def pluck(puzzle, n=0):
		"""
		Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
		in cell "c"? 
    (world model can do this..) """
		def canBeA(puz, i, j, c):
			v = puz[c//9][c%9]
			if puz[i][j] == v: return True
			if puz[i][j] in range(1,10): return False
				
			for m in range(9): # test row, col, square
				# if not the cell itself, and the mth cell of the group contains the value v, then "no"
				if not (m==c//9 and j==c%9) and puz[m][j] == v: return False
				if not (i==c//9 and m==c%9) and puz[i][m] == v: return False
				if not ((i//3)*3 + m//3==c//9 and (j//3)*3 + m%3==c%9) and puz[(i//3)*3 + m//3][(j//3)*3 + m%3] == v:
					return False

			return True

		"""
		starts with a set of all 81 cells, and tries to remove one (randomly) at a time
		but not before checking that the cell can still be deduced from the remaining cells. """
		cells     = set(range(81))
		cellsleft = cells.copy()
		while len(cells) > n and len(cellsleft):
			cell = random.choice(list(cellsleft)) # choose a cell from ones we haven't tried
			cellsleft.discard(cell) # record that we are trying this cell

			# row, col and square record whether another cell in those groups could also take
			# on the value we are trying to pluck. (If another cell can, then we can't use the
			# group to deduce this value.) If all three groups are True, then we cannot pluck
			# this cell and must try another one.
			row = col = square = False

			for i in range(9):
				if i != cell//9:
					if canBeA(puzzle, i, cell%9, cell): row = True
				if i != cell%9:
					if canBeA(puzzle, cell//9, i, cell): col = True
				if not (((cell//9)//3)*3 + i//3 == cell//9 and ((cell//9)%3)*3 + i%3 == cell%9):
					if canBeA(puzzle, ((cell//9)//3)*3 + i//3, ((cell//9)%3)*3 + i%3, cell): square = True

			if row and col and square:
				continue # could not pluck this cell, try again.
			else:
				# this is a pluckable cell!
				puzzle[cell//9][cell%9] = 0 # 0 denotes a blank cell
				cells.discard(cell) # remove from the set of visible cells (pluck it)
				# we don't need to reset "cellsleft" because if a cell was not pluckable
				# earlier, then it will still not be pluckable now (with less information
				# on the board).

		# This is the puzzle we found, in all its glory.
		return (puzzle, len(cells))

	global percent_filled
	num_given_cells = int(81*percent_filled)

	all_results, solution = run(num_given_cells, iter=10)
	puzzle = best(all_results)

	return puzzle,solution

def savePuzzles():
	'''
	From SATNet's Sudoku code generation source, save 100,000 puzzles and solutions
		https://github.com/Kyubyong/sudoku
	'''
	puzzles = np.zeros((100000, 81), np.int32)
	solutions = np.zeros((100000, 81), np.int32)
	for i, line in enumerate(open('data/sudoku.csv', 'r').read().splitlines()[1:]):
		if i == 100000:
			break 

		puzzle, solution = line.split(",")
		for j, q_s in enumerate(zip(puzzle, solution)):
			q, s = q_s
			puzzles[i, j] = q
			solutions[i, j] = s
	puzzles = puzzles.reshape((-1, 9, 9))
	solutions = solutions.reshape((-1, 9, 9))

	puzzles_tens = torch.from_numpy(puzzles)
	solutions_tens = torch.from_numpy(solutions)
	torch.save(puzzles_tens, "satnet_puzzles_100k.pt")
	torch.save(solutions_tens, "satnet_sols_100k.pt")


def vizSatNetFile(file_name="satnet_both_0.75_filled_10000.npz"):
	file = np.load(file_name)
	puzzles = file["puzzles"]
	sols = file["solutions"]

	for i in range(5):
		print(puzzles[i])
		print(f"Num zeros {81-np.count_nonzero(puzzles[i])}")
		print(sols[i])

		print("\n")

def convertToTorch(np_satnet_file):
	'''
	Splits numpy arrays into puzzle and solution torch files
		np_satnet_file: (str) is of form f'satnet_both_{percent_filled}_filled_{num_puzzles}.npz'
	'''
	file = np.load(np_satnet_file)
	puzzles = file["puzzles"]
	sols = file["solutions"]
	puzzles_tens, sols_tens = torch.from_numpy(puzzles), torch.from_numpy(sols)
	new_filename = os.path.splitext(np_satnet_file)[0] + '.pt'
	puzzle_filename = new_filename.replace("both", "puzzle")
	sol_filename = new_filename.replace("both", "sol")
	torch.save(puzzles_tens, puzzle_filename)
	torch.save(sols_tens, sol_filename)

def genSATNetPuzzlesParallel(N, pct_filled):
	puzzles = np.zeros((N, 9, 9), np.int8)
	solutions = np.zeros((N, 9, 9), np.int8)
	global percent_filled
	percent_filled = pct_filled
	pool = Pool() #defaults to number of available CPU's
	chunksize = 10
	for ind, res in enumerate(pool.imap_unordered( generateSATNetPuzzles, range(N), chunksize)):
		puzz,sol = res
		puzzles[ind,:,:] = np.squeeze(puzz)
		solutions[ind,:,:] = np.squeeze(sol)
		if ind % 1000 == 999:
			print(".", end="", flush=True)

	np.savez(f'satnet_both_{percent_filled}_filled_{N}.npz', puzzles=puzzles, solutions=solutions)

if __name__ == "__main__":
	# generatePuzzles()
	N = 100000
	genSATNetPuzzlesParallel(N, 0.85)
	vizSatNetFile(f"satnet_both_0.85_filled_{N}.npz")
	genSATNetPuzzlesParallel(N, 0.65)
	vizSatNetFile(f"satnet_both_0.65_filled_{N}.npz")
	genSATNetPuzzlesParallel(N, 0.35) # 'hard'
	vizSatNetFile(f"satnet_both_0.35_filled_{N}.npz")

	#convertToTorch("satnet_both_0.75_filled_10000.npz")

