import numpy as np 
import sys
import os 
import torch
import logging
import re 
import matplotlib.pyplot as plt 
from omegaconf import OmegaConf

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
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
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

def action_traj_idxs_unique(action_seq: np.ndarray) -> bool:
    set_indices = set()
    for i in range(action_seq.shape[0]):
        i_idx, j_idx, digit = actionToActionTuple(action_seq[i])
        set_indices.add((i_idx, j_idx))
    
    is_unique = len(list(set_indices)) == action_seq.shape[0]
    if not is_unique:
        print(f"Non-unique sequence of action indices: only {len(list(set_indices))} unique out of {action_seq.shape[0]}")
    
    return is_unique 

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def save_loss_graph(logger_file):
    '''
    Saves loss graph of the training loss and evaluation loss over timesteps in the same folder as logger_file
    '''
    save_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "loss_graph.png")
    
    with open(logger_file, 'r') as f:
        content = f.read()


    training_loss_pattern = r"training_loss: (\d+\.\d+e\+\d+)"
    evaluation_loss_pattern = r"evaluation_loss: (\d+\.\d+e\+\d+)"

    evaluation_losses = np.array([float(x) for x in re.findall(evaluation_loss_pattern, content)])
    training_losses = np.array([float(x) for x in re.findall(training_loss_pattern, content)][:len(evaluation_losses)*2])
    

    #TODO: change to extract timestep from log 
    time_steps = np.arange(0, 50*(len(training_losses)), 50)
    eval_time_steps = np.arange(0, 50*(len(training_losses)), 100)
    
    #Filter out outliers 
    training_outlier_idxs = is_outlier(training_losses)
    evaluation_outlier_idxs = is_outlier(evaluation_losses)

    plt.plot(time_steps[~training_outlier_idxs], training_losses[~training_outlier_idxs], 'b-', label='Training Loss')
    plt.plot(eval_time_steps[~evaluation_outlier_idxs], evaluation_losses[~evaluation_outlier_idxs], 'g-', label='Evaluation Loss')
    plt.legend()

    plt.xlabel("Timestep")
    plt.ylabel("Score loss")
    plt.savefig(save_file_path)

def load_config_from_run(load_dir='./'):
    cfg_path = os.path.join(load_dir, "configs/normal_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


if __name__ == "__main__":
    #save_loss_graph('experiments/09-04-2024-15:34/logs')
    save_loss_graph('experiments/10-18-2024-12:22/logs')