'''
Used to work with https://github.com/NM512/dreamerv3-torch/tree/main/envs dreamer code 
'''
import sys 
from pathlib import Path  
from sudoku_env import SudokuEnv
import gym 
import gymnasium 
import torch 
from gymnasium import spaces
from gymnasium.spaces import Discrete
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sudoku_gen import FasterSudoku, LoadSudoku
import torch 
from torch.nn import functional as F
from torch import distributions as torchd

'''
DreamerSudokuEnv is non-masked and works with dreamer code.
NMSudoku is a copied version of NoMaskSudoku which works with stable baselines/gymnasium
OneHotMask is masked wrapper that works with dreamer code.
'''

class NoMaskSudoku(gym.Env):
    '''
    Sudoku to be used with no action masking. Assumes that generated puzzle is solvable.
    Note: Does not work by itself, since action/obs space not defined or step function
    '''
    def __init__(self, n_blocks: int, percent_filled: float, puzzles_file="satnet_puzzles_100k.pt"):
        self.n_blocks = n_blocks
        self.percent_filled = percent_filled 
        self.board_width = self.n_blocks**2
        # board consists of digits and 0 (for empty cells). Board positions i,j are 0-indexed. 
        # Note that observation is a flattened vector
        self._observation_space = gym.spaces.Box(low=0,high=self.board_width, shape=(self.board_width**2,), dtype=np.uint8)
        self._action_space = gym.spaces.Discrete(self.board_width**3)
       
        self.puzzles_list = torch.load(puzzles_file)
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
        sudoku = FasterSudoku(self.board_width, self.percent_filled)
        self.sudoku = sudoku
        self.sudoku.fillValues()


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
    


    def _step(self, action: int):
        '''
        action: (int) Later converted to a tuple (i,j,act) where i,j represent board indices and act 
            represents digit placed 
        
        Returns board, is_done, reward
        '''
        (i,j,digit) = self._actionToActionTuple(action)
        self.sudoku.makeMove(i,j,digit)

        # reward is 1 if win, -1 if lose, 0 if empty cell exist
        reward = self.getReward()
        is_done = (reward != 0)

        return self.sudoku.mat.flatten(), reward, is_done, False, {}  
    
    def reset(self):
        self._setupGame()
        return self.sudoku.mat.flatten(), {} 


class DreamerSudokuEnv(NoMaskSudoku):
    """
    Written to work with https://github.com/NM512/dreamerv3-torch/tree/main/envs
    """
    def __init__(self, config):
        n_blocks = config.get("n_blocks", 3)
        percent_filled = config.get("percent_filled", 0.75)
        puzzles_file = config.get("puzzles_file", "break.pt")

        super().__init__(n_blocks, percent_filled, puzzles_file)
        self._skip_env_checking = False

    @property
    def observation_space(self):
        spaces = {
            "board": gym.spaces.Box(low=0,high=self.board_width, shape=(self.board_width**2,), dtype=np.uint8),
            "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._action_space
        action_space.discrete = True
        return action_space

    def reset(self):
        board, info = super().reset()
        obs = {
            "board": board, 
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs 
    
    def step(self, action):
        board, reward, done, _, info = self._step(action)
        reward = np.float32(reward)
        obs = {
            "board": board,
            "is_first": False,
            "is_last": done,
            "is_terminal": done
        }
        return obs, reward, done, info
    

class NMSudoku(gymnasium.Env):
    '''
    Gymansium code to work with stable-baselines
    Sudoku to be used with no action masking. Assumes that generated puzzle is solvable
    '''
    def __init__(self, env_config):

        self.n_blocks = env_config["n_blocks"]
        self.percent_filled = env_config["percent_filled"] 
        self.board_width = self.n_blocks**2
        # board consists of digits and 0 (for empty cells). Board positions i,j are 0-indexed. 
        # Note that observation is a flattened vector
        self.observation_space = gymnasium.spaces.Box(low=0,high=self.board_width, shape=(self.board_width**2,), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(self.board_width**3)
       
        self.puzzles_list = torch.load(env_config["puzzles_file"])
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
        sudoku = FasterSudoku(self.board_width, self.percent_filled)
        self.sudoku = sudoku
        self.sudoku.fillValues()


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
        (i,j,digit) = self._actionToActionTuple(action)
        self.sudoku.makeMove(i,j,digit)

        # reward is 1 if win, -1 if lose, 0 if empty cell exist
        reward = self.getReward()
        is_done = (reward != 0)

        return self.sudoku.mat.flatten(), reward, is_done, False, {}
    
    def reset(self, seed=0):
        self._setupGame()
        return self.sudoku.mat.flatten(), {} 

class MaskSudoku(gym.Env):
    '''
    Not to be used directly by itself. See DreamerMaskEnv
    Manually implemented action masking, uses gym versus gymansium for dreamer.
    is_eval: (bool) If dreamer is in eval mode, then step (in wrapper OneHotMask) returns mode() of logits versus sample()   
    is_image: (bool) If true, then board obs is now an 2d matrix. 
    '''
    def __init__(self, n_blocks: int, percent_filled: float, puzzles_file="satnet_puzzles_100k.pt", is_eval: bool = False, is_image: bool = False):
        self.n_blocks = n_blocks
        self.percent_filled = percent_filled 
        self.board_width = self.n_blocks**2
        self.is_eval = is_eval
        self.is_image = is_image 
        # board consists of digits and 0 (for empty cells). Board positions i,j are 0-indexed. 
        # Note that observation is a flattened matrix
        self._observation_space = spaces.Box(low=0,high=self.board_width, shape=(self.board_width**2,), dtype=np.uint8)
        self._action_space = spaces.Discrete(self.board_width**3, start=0)
       
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
        sudoku = FasterSudoku(self.board_width, self.percent_filled)
        self.sudoku = sudoku
        # assumes that generated sudoku puzzle is solvable
        self.sudoku.fillValues()
        self.getActionMask()
        

    def getActionMask(self):
        '''
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
    

    def _step(self, action: int):
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

class DreamerMaskEnv(MaskSudoku):
    """
    Written to work with https://github.com/NM512/dreamerv3-torch/tree/main/envs
    """
    def __init__(self, config):
        n_blocks = config.get("n_blocks", 3)
        percent_filled = config.get("percent_filled", 0.75)
        puzzles_file = config.get("puzzles_file", "break.pt")
        is_eval = config.get("is_eval", False)

        super().__init__(n_blocks, percent_filled, puzzles_file, is_eval)
        self._skip_env_checking = False

    @property
    def observation_space(self):
        spaces = {
            "board": gym.spaces.Box(low=0,high=self.board_width, shape=(self.board_width**2,), dtype=np.uint8),
            "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._action_space
        action_space.discrete = True
        return action_space

    def reset(self):
        board, info = super().reset()
        obs = {
            "board": board, 
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs 

    def step(self, action):
        board, reward, done, _, info = self._step(action)
        reward = np.float32(reward)
        obs = {
            "board": board,
            "is_first": False,
            "is_last": done,
            "is_terminal": done
        }
        return obs, reward, done, info



class OneHotMask(gym.Wrapper):
    '''
    Wrapper to be wrapped on DreamerMaskEnv- uses mask vector to generate new logits to sample from 
    '''
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gymnasium.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        self.shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=self.shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        '''
        action: (np.array) logits from OneHotDist. TODO: check that this accurate matches OneHotDist in tools.py
        If self.env.is_eval True, then return mode() rather than sample() from logits
        '''
        def mode(logits):
            _mode = F.one_hot(
                torch.argmax(logits, axis=-1), logits.shape[-1]
            )
            return _mode.detach() + logits - logits.detach()

        def sample(logits):
            # note: If there is a shape error, it's becauase the logits shape doesn't dictate the 
            # sample_shape as in OneHotDist in tools.py
           
            cat = torchd.one_hot_categorical.OneHotCategorical(logits=logits)
            sample = cat.sample() #TODO: Only works for 1d input logits tensors
            sample += logits - logits.detach()
            return sample
    
        # replace all masked locations to have very negative value
        logits = torch.tensor(action).masked_fill(torch.tensor(self.action_mask == 0), -1e8) 
        #print(f"Masked logits: {logits}")

        if self.env.is_eval:
            onehot_act = mode(logits)
        else:
            onehot_act = sample(logits)
        
        index = int(np.argmax(onehot_act))

        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        # choose a random index that is a valid action element
        indices = torch.nonzero(self.action_mask == 1).squeeze()
        random_index = indices[torch.randint(len(indices), size=(1,))]

        reference = np.zeros(actions, dtype=np.float32)
        reference[random_index] = 1.0
        return reference


class StableMaskEnv(MaskSudoku):
    '''
    MaskSudoku for stable baselines PPO. Unlike dreamer, doesn't have weird booleans and observation is a vector, not a dict
    '''
    def __init__(self, config):
        n_blocks = config.get("n_blocks", 3)
        percent_filled = config.get("percent_filled", 0.75)
        puzzles_file = config.get("puzzles_file", "break.pt")
        is_eval = config.get("is_eval", False)
        is_image = config.get("is_image", False)

        super().__init__(n_blocks, percent_filled, puzzles_file, is_eval, is_image)
        self._skip_env_checking = False

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0,high=self.board_width, shape=(self.board_width**2,), dtype=np.uint8)

    @property
    def action_space(self):
        action_space = self._action_space
        action_space.discrete = True
        return action_space
    
    def reset(self):
        board, info = super().reset()
        return board

    def step(self, action):
        board, reward, done, _, info = self._step(action)
        reward = np.float32(reward)
       
        return board, reward, done, info


