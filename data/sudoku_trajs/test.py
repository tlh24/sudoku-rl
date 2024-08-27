import numpy as np 
import logging 
from utils import check_if_valid_state_traj, check_if_solved

        
        




if __name__ == "__main__":
    file = 'forward/action/10000_trajs_yes_head.npy'
    something = np.load(file)
    print(f"Solved right: {check_if_solved(something)}")
