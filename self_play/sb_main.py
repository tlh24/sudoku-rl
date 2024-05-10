from dreamer_sudoku_env import NMSudoku
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


def main():
    env_config = {"n_blocks": 3, "percent_filled": 0.95, "puzzles_file": "/home/justin/Desktop/Code/sudoku-rl/satnet_puzzle_0.95_filled_10000.pt"}
    env = NMSudoku(env_config)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        #vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()




if __name__ == "__main__":
    main()