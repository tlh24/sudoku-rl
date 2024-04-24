import cProfile
import argparse 
import os
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import ray
from ray.rllib.algorithms import ppo, algorithm
from action_mask_env import ActionMaskEnv
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    TorchActionMaskRLM
)
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.logger import pretty_print
from sudoku_env import SudokuEnv


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop_iters", type=int, default=10, help="Number of iterations to train"
    )
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--percent_filled", type=float, default=0.75)
    parser.add_argument("--puzzles_file", type=str, default="satnet_puzzle_0.75_filled_10000.pt")
    parser.add_argument("--num_gpus",type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default='')
    args = parser.parse_args()
    return args

def training(args, algo):
    # manual training loop
    for _ in range(args.stop_iters):
        result = algo.train()
        print(pretty_print(result))

def testing(config, algo):
    # manual test loop
    print("Test loop")

    # Note: no timestep horizon limit because number of actions in sudoku game are bounded if each action is digit placed
    env = ActionMaskEnv(config["env_config"])

    obs, info = env.reset()
    done = False 

    while not done: 
        action = algo.compute_single_action(obs)
        next_obs, reward, done, truncated, _ = env.step(action)

        #print(f"Obs: {obs} Action: {action}")
        obs = next_obs 

def main():
    args = get_cli_args()    
    print(f"Running with args {args}")

    ray.init() 
    rlm_class = TorchActionMaskRLM
    rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class)

    # Use PPO action masking
    config = (
        ppo.PPOConfig()
        .environment(
            ActionMaskEnv,
            env_config={
                "n_blocks": args.n_blocks,
                "percent_filled": args.percent_filled,
                "puzzles_file" : args.puzzles_file
            }
        )
        .experimental(
            _enable_new_api_stack=True,
            _disable_preprocessor_api=True
        )
        .framework("torch")
        .resources(
            num_gpus=args.num_gpus
        )
        .rl_module(rl_module_spec=rlm_spec)
    )

    algo = config.build()

    if args.checkpoint_path:
        algo.restore(args.checkpoint_path)
    

    training(args, algo)
    # save a checkpoint 
    save_result = algo.save()
    path_to_checkpoint = save_result.checkpoint.path 
    print(f"A checkpoint has been saved at {path_to_checkpoint}")

    
    testing(config, algo)
    
    ray.shutdown()
    


if __name__ == "__main__":
    main()