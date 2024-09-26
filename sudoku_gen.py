import random
import math
import numpy as np
from termcolor import colored
import pdb
import torch
import time  
import random, copy

# TODO: Improve puzzle generation speed with https://github.com/Kyubyong/sudoku

class Sudoku:
	def __init__(self, N, K):
		self.N = N # board width, ex: 9
		self.K = K # number of cells to remove
		# Compute square root of N
		SRNd = math.sqrt(N)
		self.SRN = int(SRNd)
		self.mat = np.zeros((N, N), dtype=np.int32)
		
	def fillValues(self):
		'''
		Fills the sudoku matrix and leaves K digits empty
		'''

		# Fill the diagonal of SRN x SRN matrices
		# 4x4 matrices have a good chance of being unsolvable.
		self.mat = np.zeros((self.N, self.N), dtype=np.int32) #add this so we can properly fill values if we need to redo
		done = False
		while not done: 
			self.fillDiagonal()
			# Fill remaining blocks
			done = self.fillRemaining(0, self.SRN)
		# Remove Randomly K digits to make game
		self.removeKDigits()
		# print(self.mat)
	
	def fillDiagonal(self):
		'''
		Fills the block diagonal with blocks of width self.SRN
		'''
		for i in range(0, self.N, self.SRN):
			self.fillBoxS(i, i)
	
	def unUsedInBox(self, rowStart, colStart, num):
		for i in range(self.SRN):
			for j in range(self.SRN):
				if self.mat[rowStart + i, colStart + j] == num:
					return False
		return True
		
	def fillBoxS(self, row, col):
		'''
		Fills a box of width and height self.SRN randomly with unique numbers in [1,N]
		row, col: (int). Represent the (i,j) element offset which defines the top left corner
		of the box to be filled. 
		'''
		# just a permutation. 
		nums = [i+1 for i in range(self.N)]
		random.shuffle(nums)
		for i in range(self.SRN): 
			for j in range(self.SRN): 
				num = nums[i*self.SRN + j]
				self.mat[row + i, col + j] = num

	def fillBox(self, row, col):
		'''
		DEPRECATED. fillBoxS() is faster version.

		Fills a box of width and height self.SRN randomly with unique numbers in [1,N]
		row, col: (int). Represent the (i,j) element offset which defines the top left corner
		of the box to be filled. 
		'''
		num = 0
		for i in range(self.SRN):
			for j in range(self.SRN):
				while True:
					num = self.randomGenerator(self.N)
					if self.unUsedInBox(row, col, num):
						break
				self.mat[row + i, col + j] = num
	
	def randomGenerator(self, num):
		'''
		Returns a uniformly random number in range [1,num] inclusive
		'''
		return math.floor(random.random() * num + 1)
	
	def checkIfSafe(self, i, j, num):
		return (self.unUsedInRow(i, num) 
			 and self.unUsedInCol(j, num) 
			 and self.unUsedInBox(i - i % self.SRN, j - j % self.SRN, num))
	
	def unUsedInRow(self, i, num):
		for j in range(self.N):
			if self.mat[i,j] == num:
				return False
		return True
	
	def unUsedInCol(self, j, num):
		for i in range(self.N):
			if self.mat[i,j] == num:
				return False
		return True
		
	def checkIfValid(self): 
		# verify that the current puzzle has no contradictions. 
		valid = True
		for i in range(self.N): 
			match = self.mat == i+1
			if np.max(np.sum(match, 1)) > 1: 
				valid = False
			if np.max(np.sum(match, 0)) > 1: 
				valid = False
			blocks = []
			for i in range(0, self.N, self.SRN):
				for j in range(0, self.N, self.SRN):
					block = match[i:i+3, j:j+3].flatten()
					blocks.append(block)
			match = np.array(blocks)
			if np.max(np.sum(match, 1)) > 1: 
				valid = False
		return valid
			
	def checkOpen(self, i, j): 
		# check if there are open moves for this row, column, block
		ok = np.ones((10,))
		ok[0] = 0
		m = self.SRN
		for k in range(self.N):
			ok[self.mat[i,k]] = 0
			ok[self.mat[k,j]] = 0
			bi = i // m
			bj = j // m
			ok[self.mat[bi*m+k//m,bj*m+k%m]] = 0
		return np.sum(ok) > 0
	
	def fillRemaining(self, i, j):
		# Check if we have reached the end of the matrix
		if i == self.N - 1 and j == self.N:
			return True
	
		# Move to the next row if we have reached the end of the current row
		if j == self.N:
			i += 1
			j = 0
	
		# Skip cells that are already filled
		if self.mat[i,j] != 0:
			return self.fillRemaining(i, j + 1)
	
		# Try filling the current cell with a valid value
		nums = [i+1 for i in range(self.N)]
		random.shuffle(nums)
		for num in nums:
			if self.checkIfSafe(i, j, num):
				self.mat[i,j] = num
				# recursive -- allows for backtracking
				if self.fillRemaining(i, j + 1):
					return True
				self.mat[i,j] = 0
		
		# No valid value was found, so backtrack
		return False

	def removeKDigits(self):
		count = self.K
		while (count != 0):
			i = self.randomGenerator(self.N) - 1
			j = self.randomGenerator(self.N) - 1
			if (self.mat[i,j] != 0):
				count -= 1
				self.mat[i,j] = 0
		return
	
	def makeMove(self, i, j, num):
		self.mat[i, j] = num
		
	def takeOneStep(self): 
		# given  the current map, take one trivial elimination step.  
		out = np.array(self.mat) # deep copy
		changes = 0
		for i in range(self.N):
			for j in range(self.N): 
				if self.mat[i,j] == 0: 
					possible = np.ones((self.N,))
					for k in range(self.N):
						possible[k] = self.checkIfSafe(i, j, k+1)
					if np.sum(possible) == 1:
						out[i,j] = np.argmax(possible) + 1
						changes = changes+1
		return out,changes
		
	def hiddenSingles(self): 
		out = np.array(self.mat) # deep copy
		poss = np.ones((self.N,self.N,self.N), dtype=np.int8)
		changes = 0
		for i in range(self.N):
			for j in range(self.N): 
				v = self.mat[i,j]
				if v > 0: 
					poss[i,:,v-1] = 0
					poss[:,j,v-1] = 0
					ii = (i//3) * 3
					jj = (j//3) * 3
					poss[ii:ii+3,jj:jj+3,v-1] = 0
					poss[i,j,:] = 0
		# search for hidden singles in the 27 sets
		for v in range(self.N):
			for i in range(self.N):
				if np.sum(poss[i,:,v]) == 1: 
					j = np.argmax(poss[i,:,v]).item()
					out[i,j] = v+1
					changes = changes+1
			for j in range(self.N):
				if np.sum(poss[:,j,v]) == 1: 
					i = np.argmax(poss[:,j,v]).item()
					out[i,j] = v+1
					changes = changes+1
			for b in range(self.N): 
				ii = (b//3)*3
				jj = (b%3)*3
				if np.sum(poss[ii:ii+3,jj:jj+3,v]) == 1: 
					k = np.argmax(poss[ii:ii+3,jj:jj+3,v].reshape((9,))).item()
					ki = k//3
					kj = k%3
					out[ii+ki,jj+kj] = v+1
					changes = changes+1
		return out,changes

	def setMat(self, mat): 
		self.mat = mat.astype(np.int32) # must be int!  == comparisons! 

	def printSudoku(self, indent, puzzl_mat, guess_mat=None, curs_pos=None):
		for i in range(self.N):
			print(indent, end="")
			for j in range(self.N):
				k = i // self.SRN + j // self.SRN
				color = "black" if k % 2 == 0 else "red"
				p = int(puzzl_mat[i,j])
				if guess_mat is not None:
					if np.floor(guess_mat[i,j]) > 0:
						p = int(guess_mat[i,j])
						color = "blue" if k % 2 == 0 else "magenta"
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
		self.mat = puzzl_mat
		if guess_mat is not None: 
			self.mat = self.mat + guess_mat
		print(f"{indent}Valid:", self.checkIfValid())
		# print the cannonical form
		for i in range(self.N):
			for j in range(self.N):
				print(puzzl_mat[i,j], end="")
		print(",")


def generateInitialBoard(percent_filled=0.75, exact_num_filled=False):
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


	def run(num_given_cells, iter=1, match_num_given=True):
		'''
		Attempts to create a puzzle with num_given_cells,
		but if it can't returns puzzle
		as close to that and larger in set of iter puzzles generated
		If match_num_given is true, then only returns a puzzle that has exactly num_given_cell digits
		'''
		all_results = {}
	#     print "Constructing a sudoku puzzle."
	#     print "* creating the solution..."
		a_puzzle_solution = construct_puzzle_solution()

		if match_num_given:
			num_times = 0
			while True: 
				print(f"Num times {num_times}")
				puzzle = copy.deepcopy(a_puzzle_solution)
				(result, number_of_cells) = pluck(puzzle, num_given_cells)
				print(f'Number of cells {number_of_cells} vs number given {num_given_cells}')
				if number_of_cells == num_given_cells:
					all_results[num_given_cells] = [result]
					break 
				num_times += 1

			return all_results, a_puzzle_solution
		
		#print "* constructing a puzzle..."
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

	def pluck(puzzle, num_given_cells):
		"""
		Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
		in cell (ii,jj)? """
		def canBeA(puz, i, j, ii, jj):
			v = puz[ii][jj]
			if puz[i][j] == v: return True
			if puz[i][j] in range(1,10): return False
				
			for m in range(9): # test row, col, square
				# if not the cell itself, and the mth cell of the group contains the value v, then "no"
				if not (m==ii and j==jj) and puz[m][j] == v: return False
				if not (i==ii and m==jj) and puz[i][m] == v: return False
				if not ((i//3)*3 + m//3==ii and (j//3)*3 + m%3==jj) and puz[(i//3)*3 + m//3][(j//3)*3 + m%3] == v:
					return False

			return True


		"""
		starts with a set of all 81 cells, and tries to remove one (randomly) at a time
		but not before checking that the cell can still be deduced from the remaining cells. """
		cells     = set(range(81))
		cellsleft = cells.copy()
		while len(cells) > num_given_cells and len(cellsleft):
			cell = random.choice(list(cellsleft)) # choose a cell from ones we haven't tried
			cellsleft.discard(cell) # record that we are trying this cell
			# row, col and square record whether another cell in those groups could also take
			# on the value we are trying to pluck. (If another cell can, then we can't use the
			# group to deduce this value.) If all three groups are True, then we cannot pluck
			# this cell and must try another one.
			row = col = block = False
			ii = cell // 9 # row
			jj = cell % 9 # column
			for k in range(9):
				if k != ii: # iterate over a column
					if canBeA(puzzle, k, jj, ii, jj): col = True
				if k != jj: # iterate over a row
					if canBeA(puzzle, ii, k, ii, jj): row = True
				bi = ((ii)//3)*3 # block row
				bj = ((jj)%3)*3 # block col
				if not (bi + k//3 == ii and bj + k%3 == jj):
					if canBeA(puzzle, bi + k//3, bj + k%3, ii, jj): block = True

			if row and col and block:
				# this means that trivial deduction cannot determine the value in the cell -- there is another option of the same value in this cell's row, column, or block. So, try again.
				continue 
			else:
				# this is a pluckable cell!
				puzzle[cell//9][cell%9] = 0 # 0 denotes a blank cell
				cells.discard(cell) # remove from the set of visible cells (pluck it)
				# we don't need to reset "cellsleft" because if a cell was not pluckable
				# earlier, then it will still not be pluckable now (with less information
				# on the board).

		# This is the puzzle we found, in all its glory.
		return (puzzle, len(cells))


	num_given_cells = int(81*percent_filled)
	
	all_results, solution = run(num_given_cells, iter=10, match_num_given=exact_num_filled)
	quiz = best(all_results)
	return quiz
		
class FasterSudoku(Sudoku):
	'''
	Sudoku class that generates a puzzle following the satnet puzzle generation code
	'''
	def __init__(self, N, percent_filled):
		super().__init__(N,0)
		self.percent_filled = percent_filled 
	
	def fillValues(self):
		'''
		Populates the board and generates an initial board
		'''
		self.mat = np.array(generateInitialBoard(self.percent_filled), dtype=np.int32)


class LoadSudoku(Sudoku):
	'''
	Sudoku class but "generates" a puzzle by sampling from a file containg a list of puzzles
	'''
	def __init__(self, N, puzzles=None, file_name="puzzles_500000.pt"):
		super().__init__(N,0)
		if puzzles is None:
			self.puzzles_list = torch.load(file_name)
		else:
			self.puzzles_list = puzzles 
		assert N == self.puzzles_list.shape[1]

	def fillValues(self):
		#generates the board by choosing a random init board from list of puzzles
		num_boards = self.puzzles_list.size(0)
		rand_idx = torch.randint(num_boards, size=(1,)).item()
		# create a copy to prevent changing the original puzzles list
		rand_board = self.puzzles_list[rand_idx].clone().detach().numpy()
		assert rand_board.shape == (self.N, self.N)
		self.mat = rand_board.astype(np.uint8)


# Driver code
if __name__ == "__main__":
	N = 9
	K = 81-37
	if False: # need to get the datafile from Justin
		#sudoku = LoadSudoku(N, K)
		#sudoku.fillValues()
		#sudoku.printSudoku()
		puzzles_file = "satnet_puzzle_0.95_filled_10000.pt"
		puzzles_list = torch.load(puzzles_file)
		start = time.perf_counter()
		for _ in range(1000):
			sudoku = FasterSudoku(9, 0.75)
			sudoku.fillValues()
		end = time.perf_counter()
		elapsed = end-start 
		print(f"Faster sudoku took: {elapsed}s for 1000 iters")

		start = time.perf_counter()
		for _ in range(1000):
			sudoku = LoadSudoku(9, puzzles_list)
			sudoku.fillValues()
		end = time.perf_counter()
		elapsed = end-start 
		print(f"Load sudoku took: {elapsed}s for 1000 iters")

	sudoku = Sudoku(N, K)
	sudoku.fillValues()
	sudoku.printSudoku("", sudoku.mat)
	print("base",sudoku.checkIfValid())
	# check the validate fn. 
	for r in range(N): 
		for c in range(N): 
			if sudoku.mat[r,c] == 0: 
				for i in range(N):
					sudoku.mat[r,c] = i+1
					print(r,c,i+1,sudoku.checkIfValid())
				sudoku.mat[r,c] = 0

	# check the open Fn
	sudoku.printSudoku("", sudoku.mat)
	for r in range(N): 
		for c in range(N): 
			if sudoku.mat[r,c] == 0: 
				print(r,c,sudoku.checkOpen(r,c))
				
	# check FasterSudoku
	sudokuf = FasterSudoku(N, 25/81)
	sudokuf.fillValues()
	sudokuf.printSudoku("", sudokuf.mat)
	print("")

	# check original Sudoku (which frequently admits more than one solution)
	sudoku = Sudoku(N, 81-25)
	sudoku.fillValues()
	sudoku.printSudoku("", sudoku.mat)
	step,changes = sudoku.takeOneStep()
	print(f"\none step: {changes} changes")
	sudoku.printSudoku("", step)
