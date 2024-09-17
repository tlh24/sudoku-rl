import numpy as np 
import logging 
from utils import check_if_valid_state_traj, check_if_solved, action_traj_idxs_unique

        





if __name__ == "__main__":
    file = 'forward/action/10000_trajs_0.45_filled_yes_head.npy'
    action_trajs = np.load(file)
    for i in range(len(action_trajs)):
        is_unique = action_traj_idxs_unique(action_trajs[i])
        if not is_unique:
            print("error; traj not unique")
