import random
import math
import numpy as np
from termcolor import colored
import pdb
import torch 

# TODO: Improve puzzle generation speed with https://github.com/norvig/pytudes/blob/main/ipynb/Sudoku.ipynb

class Sudoku:
	def __init__(self, N, K):
		self.N = N # board width, ex: 9
		self.K = K # number of cells to remove
		# Compute square root of N
		SRNd = math.sqrt(N)
		self.SRN = int(SRNd)
		self.mat = np.zeros((N, N), dtype=np.int32)
		

	
	def fillValues(self):
		# Fill the diagonal of SRN x SRN matrices
		# 4x4 matrices have a good chance of being unsolvable.
		done = False
		while not done: 
			self.fillDiagonal()
			# Fill remaining blocks
			done = self.fillRemaining(0, self.SRN)
		# Remove Randomly K digits to make game
		self.removeKDigits()
		print(self.mat)
	
	def fillDiagonal(self):
		for i in range(0, self.N, self.SRN):
			self.fillBoxS(i, i)
	
	def unUsedInBox(self, rowStart, colStart, num):
		for i in range(self.SRN):
			for j in range(self.SRN):
				if self.mat[rowStart + i, colStart + j] == num:
					return False
		return True
		
	def fillBoxS(self, row, col):
		# just a permutation. 
		nums = [i+1 for i in range(self.N)]
		random.shuffle(nums)
		for i in range(self.SRN): 
			for j in range(self.SRN): 
				num = nums[i*self.SRN + j]
				self.mat[row + i, col + j] = num
	
	def fillBox(self, row, col):
		num = 0
		for i in range(self.SRN):
			for j in range(self.SRN):
				while True:
					num = self.randomGenerator(self.N)
					if self.unUsedInBox(row, col, num):
						break
				self.mat[row + i, col + j] = num
	
	def randomGenerator(self, num):
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

	def getLegalMoveMask(self):
		'''
		Updates action_mask to be a binary vector where 1 for each action that is a valid move, 0 for an action 
			that is invalid. See SudokuEnv for more context- the action space is a int in [0, board_width**3) which 
			maps to a (i, j, digit) where i,j is the board cell location and digit is the proposed digit placement 
		'''


	def setMat(self, mat): 
		self.mat = mat.astype(np.int32) # must be int!  == comparisons! 

	def printSudoku(self):
		for i in range(self.N):
			for j in range(self.N):
				k = i // self.SRN + j // self.SRN
				color = "black" if k % 2 == 0 else "red"
				p = math.floor(self.mat[i,j])
				print(colored(p, color), end=" ")
			print()


class LoadSudoku(Sudoku):
	'''
	Sudoku class but "generates" a puzzle by sampling from a file containg a list of puzzles
	'''
	def __init__(self, N, K, puzzles=None, file_name="puzzles_500000.pt"):
		super().__init__(N,K)
		if puzzles is None:
			self.puzzles_list = torch.load(file_name)
		else:
			self.puzzles_list = puzzles 
		assert N == self.puzzles_list.shape[1]

	def fillValues(self):
		#generates the board by choosing a random init board from list of puzzles
		num_boards = self.puzzles_list.size(0)
		rand_idx = torch.randint(num_boards, size=(1,)).item()
		rand_board = self.puzzles_list[rand_idx].numpy()
		assert rand_board.shape == (self.N, self.N)
		self.mat = rand_board



# Driver code
if __name__ == "__main__":
	N = 9
	K = 81-37
	sudoku = LoadSudoku(N, K)
	sudoku.fillValues()
	sudoku.printSudoku()




