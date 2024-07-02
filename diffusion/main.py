from utils import set_seed, Trainer, TrainerConfig 
import argparse
import torch 
from torch.utils.data import DataLoader
import gym 
from datasets.data import SequenceDataset 
from model import TemporalUnet, GaussianDiffusion
import pdb 

def main(args=None):
    seed = 42
    ###
    #Load Data
    ###
    env = gym.make('maze2d-large-v1')
    set_seed(seed, env)

    # TODO: add more wrappers from https://github.com/ikostrikov/implicit_q_learning/blob/master/wrappers/episode_monitor.py
    #env = wrappers.EpisodeMonitor(env)
    #env = wrappers.SinglePrecision(env)
    dataset = SequenceDataset(env_name='maze2d-large-v1', horizon=64)

    ###
    #Build diffusion model and trainer
    ###
    #TODO: figure out how the replanner can make the unet of size action yet noise the observation?? 
    # rely on janner code instead
    #unet = TemporalUnet(horizon=args.H, )


    ###
    #Train model
    ###
    train_config = TrainerConfig()
    trainer = Trainer(model, dataset, train_config)
    for trajectories in dataloader:
        print(trajectories)
        break 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128, help='Horizon length')
    parser.add_argument('--dsteps', type=int, default=400, help='Num diffusion steps')
    args = parser.parse_args()
    main(args)

