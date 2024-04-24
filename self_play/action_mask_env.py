from sudoku_env import SudokuEnv
import gymnasium as gym 
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
import numpy as np
from collections import OrderedDict


class ActionMaskEnv(SudokuEnv):
    """
    Adapted from https://github.com/ray-project/ray/blob/master/rllib/examples/envs/classes/action_mask_env.py
    Publishes an action mask at each step for our SudokuEnv
    """

    def __init__(self, config):
        n_blocks = config.get("n_blocks", 3)
        percent_filled = config.get("percent_filled", 0.75)
        puzzles_file = config.get("puzzles_file", "break.pt")
      
        super().__init__(n_blocks, percent_filled, puzzles_file)
        self._skip_env_checking = False 
        # Masking only works for Discrete actions.
        assert isinstance(self.action_space, Discrete)
        # Add action_mask to observations
        self.observation_space = Dict(
            {
                "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,)),
                "observations": self.observation_space,
            }
        )

    def reset(self, *, seed=None, options=None):
        board, info = super().reset()
        obs = OrderedDict([
            ("action_mask", self.action_mask),
            ("observations", board)
        ])
        return obs, info

    def step(self, action: int):
        if not self.action_mask[action]:
            raise ValueError(
                f"Invalid action ({action}) sent to env! "
                f"Board={self.sudoku.mat}"
                f"action_mask={self.action_mask}"
            )
        board, rew, done, _, _ = super().step(action)
                
        obs = OrderedDict([
            ("action_mask", self.action_mask),
            ("observations", board)
        ])

        return obs, rew, done, False, {}

    #def _get_action_mask(self, board):
    #    self.valid_actions = super().get_action_mask(board)
    #    return self.valid_actions