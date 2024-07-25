from dreamer_sudoku_env import StableMaskSudoku
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import pdb 


def main():
    config = {"n_blocks": 3, "percent_filled": 0.95, "puzzles_file": "/home/justin/Desktop/Code/sudoku-rl/satnet_puzzle_0.95_filled_10000.pt", "is_eval": False}
    n_blocks = config.get("n_blocks", 3)
    percent_filled = config.get("percent_filled", 0.75)
    puzzles_file = config.get("puzzles_file", "break.pt")
    is_eval = config.get("is_eval", False)
    
    env = StableMaskSudoku(n_blocks, percent_filled, puzzles_file, is_eval)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    breakpoint()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(f"Obs:\n {obs.reshape((9,9))}")
        print(f"reward: {reward}  action: {env._actionToActionTuple(action)}")

        
        #vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()




if __name__ == "__main__":
    main()