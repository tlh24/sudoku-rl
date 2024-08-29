import numpy as np 
import sys
from pathlib import Path 
sys.append(Path(__file__).resolve().parent.parent)
from data.sudoku_trajs.utils import actionToActionTuple

def action_seq_to_board(action_seq: np.ndarray):
    ''''
    Converts an action sequence of ints to a final board  

    action_seq: sequence of action integers (seq_len,) or (1, seq_len)
    '''
    action_seq = action_seq.flatten()
    current_state = np.zeros((9,9))
    for i in range(len(action_seq)):
        i_idx, j_idx, digit = actionToActionTuple(action_seq[i])
        current_state[i_idx][j_idx] = digit 
    
    return current_state