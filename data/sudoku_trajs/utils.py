import logging 
import numpy as np 

board_width = 9

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

def check_if_valid_state_traj(trajs: np.ndarray):
    '''
    Ensures the transition is only one digit placed at each time 
    '''
    if len(trajs.shape) == 3:
        for i in range(0, len(trajs)):
            diff = trajs[i+1] - trajs[i]
            non_zero = diff[diff!= 0]
            if len(non_zero) != 1:
               logging.error(f"Getting non-valid transition at timestep {i}; {trajs[i+1]}\n\n {trajs[i]}") 
               return False 
    if len(trajs.shape) == 4:
        for n in range(0, len(trajs)):
            for i in range(0, len(trajs[0])-1):
                diff = trajs[n][i+1] - trajs[n][i]
                non_zero = diff[diff  != 0]
                if len(non_zero) != 1:
                    breakpoint()
                    logging.error(f"Getting non-valid transition at timestep {i}; {trajs[n][i+1]}\n {trajs[n][i]}") 
                    logging.error(diff)
                    return False 
    return True    

def check_if_solved(trajs: np.ndarray):
    '''
    Given an array of trajectories, checks if the trajectory ends up returning a valid solution

    trajs: (np.ndarray) Shape (num_samples, traj_length, 9,9) or (num_samples, traj_length)
    '''
    num_samples = trajs.shape[0]
    for i in range(num_samples):
        traj = trajs[i]
        if len(traj.shape) > 1: #states trajectory
            final_board = traj[-1]
        else: #action trajectory
            final_board = np.zeros((9,9))
            for action_num in traj:
                (i,j, digit) = actionToActionTuple(action_num)
                final_board[i][j] = digit 
            assert len(final_board[final_board == 0]) == 0
        
        is_valid = isValidSudoku(final_board)
        if is_valid == False: return False
     
    return True 


def action_traj_idxs_unique(action_seq: np.ndarray) -> bool:
    '''
    Check if an action directoy makes a unique cell placement 
    '''
    set_indices = set()
    for i in range(action_seq.shape[0]):
        i_idx, j_idx, digit = actionToActionTuple(action_seq[i])
        set_indices.add((i_idx, j_idx))
    
    is_unique = len(list(set_indices)) == action_seq.shape[0]
    if not is_unique:
        print(f"Non-unique sequence of action indices: only {len(list(set_indices))} unique out of {action_seq.shape[0]}")
    
    return is_unique      
            


