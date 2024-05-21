from dreamer_sudoku_env import StableMaskSudoku, DreamerMaskEnv, OneHotMask
import pdb 
import numpy as np 


def test_board_mask():
    '''
    Checks to see if the board mask makes sense after the moves
    '''
    pass 

if __name__ == "__main__":
    config = {"n_blocks": 3, "percent_filled": 0.95, "puzzles_file": "/home/justin/Desktop/Code/sudoku-rl/satnet_puzzle_0.95_filled_10000.pt", "is_eval": False}
    n_blocks = config.get("n_blocks", 3)
    percent_filled = config.get("percent_filled", 0.75)
    puzzles_file = config.get("puzzles_file", "break.pt")
    is_eval = config.get("is_eval", False)
    env = DreamerMaskEnv(config)
    env = OneHotMask(env)

    obs = env.reset()
    # visually test if the logits are legal masked
    for  _ in range(7):
        print(f"Board:\n {env.sudoku.mat}")
        #print(f"Board mask: {env.action_mask}")
        rand_action = np.zeros(env.shape)
        env.step(rand_action)
        # print logits in step

        



        



