import numpy as np 
import sys
import os 
import torch
import logging
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sudoku_trajs'))

def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)

def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


board_width = 9
def actionTupleToAction(action_tuple):
        '''
        Converts (i,j,digit) tuple to an action integer in [0, board_width^3]
        '''
        (i,j,digit) = action_tuple

        return i * (board_width**2) + j * (board_width) + digit - 1

def actionToActionTuple(action_num:int):
    '''
    Given an action num in [0, board_width^3), convert to a tuple which represents
        (i,j, digit) where i,j in [0,board_width) and digit in [1,board_width]
    '''
    i = action_num // (board_width**2)
    remainder = action_num % (board_width**2)
    j = remainder // board_width 
    digit = (remainder % board_width) + 1

    return (i, j, digit)  

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

def isValidSudoku(board) -> bool:
    if isinstance(board, np.ndarray):
        board = board.tolist()
    
    for i in range(9):
        row = board[i]
        if len(row)!=len(set(row)): return False
        col = [board[c][i] for c in range(9)]
        if len(col)!=len(set(col)): return False
        box = [board[ind//3+(i//3)*3][ind%3+(i%3)*3] for ind in range(9)]
        if len(box)!=len(set(box)): return False
    return True
