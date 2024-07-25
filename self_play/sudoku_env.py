import sys
from pathlib import Path  
import gymnasium
from gymnasium import spaces
import numpy as np 

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sudoku_gen import Sudoku, LoadSudoku
import torch 


class SudokuEnv(gymnasium.Env):
    '''
    To be used with action masking 
    '''
    def __init__(self, n_blocks: int, percent_filled: float, puzzles_file="satnet_puzzles_100k.pt"):
        self.n_blocks = n_blocks
        self.percent_filled = percent_filled 
        self.board_width = self.n_blocks**2
        # board consists of digits and 0 (for empty cells). Board positions i,j are 0-indexed. 
        # Note that observation is a flattened matrix
        self.observation_space = spaces.Box(low=0,high=self.board_width, shape=(self.board_width**2,), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.board_width**3, start=0)
       
        self.puzzles_list = torch.load(puzzles_file)
        self.action_mask = np.zeros(self.board_width**3, dtype=np.int8)
        self._setupGame()

    def _actionTupleToAction(self, action_tuple):
        '''
        Converts (i,j,digit) tuple to an action integer in [0, board_width^3]
        '''
        (i,j,digit) = action_tuple

        return i * (self.board_width**2) + j * (self.board_width) + digit - 1


    def _actionToActionTuple(self, action_num:int):
        '''
        Given an action num in [0, board_width^3), convert to a tuple which represents
            (i,j, digit) where i,j in [0,board_width) and digit in [1,board_width]
        '''
        i = action_num // (self.board_width**2)
        remainder = action_num % (self.board_width**2)
        j = remainder // self.board_width 
        digit = (remainder % self.board_width) + 1

        return (i, j, digit)       
    
    def _setupGame(self):
        sudoku = LoadSudoku(self.board_width, self.puzzles_list)
        self.sudoku = sudoku
        num_attempts = 0
        while np.sum(self.action_mask) == 0:
            if num_attempts > 10:
                raise RuntimeError(f"Failed to get a valid board 10 times board={self.sudoku.mat}")
            self.sudoku.fillValues()
            self.getActionMask()
            num_attempts += 1
        

    def getActionMask(self):
        '''
        TODO: optimize valid action checking 
        
        Given current board state, return a boolean action mask (vector of size board_width^3) where 
            all valid elements are 1 and invalid are 0 
        
        board: (numpy matrix) Size board_width * board_width, each elm in matrix is a digit in [0, board_width]
            where 0 represents no number
        '''
        action_mask = np.ones(self.board_width**3, dtype=np.int8)
        # all occupied cells are invalid actions
        for i in range(0, self.board_width):
            for j in range(0, self.board_width):
                if self.sudoku.mat[i][j] != 0:
                    action_mask[self._actionTupleToAction((i,j,1)) : self._actionTupleToAction((i,j,self.board_width)) + 1] = 0
                else:
                    for digit in range(1, self.board_width + 1):
                        if not self.sudoku.checkIfSafe(i,j,digit):
                            action_mask[self._actionTupleToAction((i,j,digit))] = 0
        
        self.action_mask = action_mask


    def getReward(self):
        '''
        Returns 1 if game is valid solved, -1 if game is incorrectly solved or empty cells but no further playable actions,
          0 if empty cells left and playable moves
        '''
        board = self.sudoku.mat 

        row_seen_sets = [set() for _ in range(self.board_width)]
        col_seen_sets = [set() for _ in range(self.board_width)]
        box_seen_sets = [set() for _ in range(self.board_width)]

        for i in range(self.board_width):
            for j in range(self.board_width):
                val = board[i][j]
                # Empty cell exists
                if val == 0: 
                    # no playable moves left is loss
                    if sum(self.action_mask) == 0:
                        return -1 
                    return 0

                # check row 
                if val in row_seen_sets[i]:
                    return -1 
                row_seen_sets[i].add(val)
                # check col 
                if val in col_seen_sets[j]:
                    return -1 
                col_seen_sets[j].add(val) 

                # check box 
                box_idx = (i // self.n_blocks) * self.n_blocks + (j // self.n_blocks)
                if val in box_seen_sets[box_idx]:
                    return -1 
                box_seen_sets[box_idx].add(val)
        
        # If game correctly solved        
        return 1 	
    


    def step(self, action: int):
        '''
        action: (int) Later converted to a tuple (i,j,act) where i,j represent board indices and act 
            represents digit placed 
        
        Returns board, is_done, reward
        '''
        # need to limit to only legal actions
        # assuming that action is legal
        (i,j,digit) = self._actionToActionTuple(action)
        self.sudoku.makeMove(i,j,digit)

        # update my action mask
        self.getActionMask()

        # reward is 1 if win, -1 if lose/no playable actions and empty cells left, 0 if there exists an empty cell and playable 
        reward = self.getReward()
        is_done = (reward != 0)

        return self.sudoku.mat.flatten(), reward, is_done, False, {}  
    
    def reset(self):
        self._setupGame()
        return self.sudoku.mat.flatten(), {} 
    

