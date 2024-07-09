from utils import set_seed, Trainer, TrainerConfig 
import argparse
import torch 
from torch.utils.data import DataLoader
import gym 
from datasets.data import SequenceDataset 
from model import TemporalUnet, GaussianDiffusion
import pdb 
import numpy as np 
ENV_NAME = 'maze2d-large-v1'
DTYPE = torch.float
DEVICE = 'cuda'

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
	return torch.tensor(x, dtype=dtype, device=device)

def get_action(plan, t):
    '''
    plan: sequence of states from t = 0 to t_final
    Returns the action at time t by calculating the difference in the x,y location 
    of plan[t+1] - plan[t]
    '''
    #TODO: need to check the shape to make sure that you index the shit right
    plan = torch.flatten(plan)
    return plan[t+1][:2] - plan[t][:2]

def eval(trainer, load_path, device, history_len = 20,):
    #TODO: adapt the code for multiple envs and verify
    trainer.load(load_path)
    trainer.model.eval()
    env = gym.make(ENV_NAME)

    obs_history = np.array([env.reset()[None]]) #start with obs shape (1,4)
    episode_reward = 0
    finished = False 
    t = 0
    while not finished:
        #TODO: check if normalizing already normalized does nothing
        trainer.train_dataset.normalizers['observations'].normalize(obs_history)
        
        # condition on the last history_len obs
        conditions = {0: to_torch(obs_history[-history_len:], device)}
        sample = trainer.model.conditional_sample(conditions)
        unnormalized_sample = trainer.train_dataset.normalizers['observations'].unnormalize(sample)
        action = get_action(unnormalized_sample, t)

        obs, reward, done, _ = env.step(action)
        obs_history = np.append(obs_history, obs[None])
        episode_reward += reward
        if done:    
            finished = True 
            break
        t += 1

    print(f"Final episode sum reward: {episode_reward}")


def main(args=None):
    seed = 42
    ###
    #Load Data
    ###
    env = gym.make(ENV_NAME)
    set_seed(seed, env)

    # TODO: add more wrappers from https://github.com/ikostrikov/implicit_q_learning/blob/master/wrappers/episode_monitor.py
    #env = wrappers.EpisodeMonitor(env)
    #env = wrappers.SinglePrecision(env)
    #TODO: make the training horizon smaller than the inference horizon if doesn't work
    dataset = SequenceDataset(env_name=ENV_NAME, horizon=args.H)

    ###
    #Build diffusion model and trainer
    ###
    #TODO: figure out how the replanner can make the unet of size action yet noise the observation?? 
    # rely on janner code instead
    obs_dim, act_dim = 4, 2 #hard-coded for maze2d
    unet = TemporalUnet(horizon=args.H, cond_dim = None, transition_dim=obs_dim, dim = 128, dim_mults=(1,2,4,8))

    diffusion = GaussianDiffusion(unet, args.dsteps)


    ###
    #Train model
    ###
    train_config = TrainerConfig()
    trainer = Trainer(diffusion, dataset, train_config)
    if args.train:
        trainer.train()
    else:
        eval(trainer, SOMELOADPATH, device=SOMEDEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128, help='Horizon length')
    parser.add_argument('--dsteps', type=int, default=128, help='Num diffusion steps')
    parser.add_argument('--train',type=bool, default=True, help="Should train diffusion or eval")
    args = parser.parse_args()
    main(args)

