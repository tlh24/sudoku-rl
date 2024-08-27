import numpy as np 
import logging 

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
        
        




if __name__ == "__main__":
    file = 'forward/action/10000_trajs_yes_head.npy'
    something = np.load(file)
    breakpoint()
    #check_if_valid_state_traj(something)
