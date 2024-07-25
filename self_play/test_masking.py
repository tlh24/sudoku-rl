from dreamer_sudoku_env import StableMaskEnv, DreamerMaskEnv, OneHotMask
import pdb 
import numpy as np 
import gym 
from tqdm import tqdm
#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines import PPO2
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def check_random_agent():
    config = {"n_blocks": 3, "percent_filled": 0.95, "puzzles_file": "/home/justin/Desktop/Code/sudoku-rl/satnet_puzzle_0.95_filled_10000.pt", "is_eval": False}
    env = DreamerMaskEnv(config)
    env = OneHotMask(env)
    obs = env.reset()
    # visually test if the logits are legal masked
    for  _ in range(7):
        print(f"Board:\n {env.sudoku.mat}")
        rand_action = np.zeros(env.shape)
        env.step(rand_action)
        # print logits in step


def train_test_agent(agent_type: str, train_timesteps: int, num_test_episodes=100, percent_filled:int=0.75):
    '''
    agent_type: (str) Either "ppo" or "ppo_mlp" or "random".
        PPO_MLP uses a MLP which takes in a vector as observation, not a dictionary
    '''

    if agent_type == "ppo":
        # train the agent
        config = {"n_blocks": 3, "percent_filled": percent_filled, "puzzles_file": "/home/justin/Desktop/Code/sudoku-rl/satnet_puzzle_0.95_filled_10000.pt", "is_eval": False}
        train_env = DreamerMaskEnv(config)
        train_env = OneHotMask(train_env)
        obs = train_env.reset()
        train_env = make_vec_env(lambda: train_env, n_envs=4)
        model = PPO("MultiInputPolicy", train_env, verbose=1)
        model.learn(total_timesteps=train_timesteps)
    if agent_type == "ppo_mlp":
        config = {"n_blocks": 3, "percent_filled": percent_filled, "puzzles_file": "/home/justin/Desktop/Code/sudoku-rl/satnet_puzzle_0.95_filled_10000.pt", "is_eval": False}
        train_env = StableMaskEnv(config)
        train_env = OneHotMask(train_env)
        obs = train_env.reset()
        train_env = make_vec_env(lambda: train_env, n_envs=4)
        model = PPO("MlpPolicy", train_env, verbose=1)
        model.learn(total_timesteps=train_timesteps)


        
        
    # evaluate the agent
    test_config = {"n_blocks": 3, "percent_filled": percent_filled, "puzzles_file": "/home/justin/Desktop/Code/sudoku-rl/satnet_puzzle_0.95_filled_10000.pt", "is_eval": True}
    if agent_type == "ppo_mlp":
        test_env = StableMaskEnv(test_config)
    else:
        test_env = DreamerMaskEnv(test_config)
    test_env = OneHotMask(test_env) 
    test_env = make_vec_env(lambda: test_env, n_envs=1)
    eval_rewards = []
    for _ in tqdm(range(num_test_episodes)):
        obs = test_env.reset()
        done = False 
        while not done:
            if agent_type == "random":
                action = test_env.action_space.sample()
            else:
                action, _states = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
        eval_rewards.append(reward)
    
    avg_eval_reward = np.mean(eval_rewards)
    print(f"Average eval reward {avg_eval_reward}")


  

if __name__ == "__main__":
    #breakpoint()
    train_test_agent("ppo", 2000000, 1000, percent_filled=0.75)
    
    
   

        



        



