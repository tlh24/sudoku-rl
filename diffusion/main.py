from utils import set_seed 
import argparse
import torch 
from torch.utils.data import DataLoader
import gym 
from datasets.dataset import SimplifiedSequenceDataset 


def main(args=None):
    seed = 42
    ###
    #Load Data
    ###
    env = gym.make('maze2d-medium-v1')
    set_seed(seed, env)

    # TODO: add more wrappers from https://github.com/ikostrikov/implicit_q_learning/blob/master/wrappers/episode_monitor.py
    #env = wrappers.EpisodeMonitor(env)
    #env = wrappers.SinglePrecision(env)
    dataset = SimplifiedSequenceDataset(env='antmaze-medium-diverse-v2', horizon=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    ###
    #Build diffusion model and trainer
    ###

    ###
    #Train model
    ###
    for batch in dataloader:
        trajectories, conditions = batch 
        pass 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128, help='Horizon length')
    parser.add_argument('--dsteps', type=int, default=400, help='Num diffusion steps')
    args = parser.parse_args()
    main(args)

