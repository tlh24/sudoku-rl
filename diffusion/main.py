from utils import set_seed, Trainer, TrainerConfig 
import argparse
import torch 
from torch.utils.data import DataLoader
import gym 
from datasets.data import SequenceDataset 
from model import TemporalUnet, GaussianDiffusion, InvKinematicsModel
import pdb 
import numpy as np 
ENV_NAME = 'maze2d-large-v1'
DTYPE = torch.float
DEVICE = 'cuda'
NUM_ENVS = 1

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
	return torch.tensor(x, dtype=dtype, device=device)

def get_action(plan, t: int):
    '''
    plan: sequence of states from t = 0 to t_final
    
    Returns the action at time t by calculating the difference in the x,y location 
    of plan[t+1] - plan[t]
    '''
    #TODO: need to check the shape to make sure that you index the shit right
    plan = torch.flatten(plan)
    return plan[t+1][:2] - plan[t][:2]

def eval(trainer, diffusion_loadpath, ema_loadpath, inv_kin_loadpath, device, horizon, history_len=20):
    #TODO: adapt the code for multiple envs and verify
    trainer.load(diffusion_loadpath, ema_loadpath, inv_kin_loadpath)
    for name, model in trainer.models.items():
        model.eval()
    env = gym.make(ENV_NAME)
    pdb.set_trace()
    obs_history = np.array([env.reset()[None] for _ in range(NUM_ENVS)]) # shape is (num_envs, num_timesteps, state_dim)
 
    episode_reward = 0
    finished = False 
    t = 0
    while not finished:
        norm_history = trainer.train_dataset.normalizers['observations'].normalize(obs_history[-history_len:]) 
        conditions = {} # key is time step t to replace, value is state at timestep t  
        for i in range(0, len(norm_history)):
             conditions[i] = to_torch(norm_history[:, i], device)
        
        sample = trainer.models['diffusion'].conditional_sample(conditions, horizon)
        # generate action from normalized plan 
        unnormalized_sample = trainer.train_dataset.normalizers['observations'].unnormalize(sample)
        
        predicted_actions = trainer.models['inv_kinematics'](sample) #(batch*H, act_dim)
        horizon, batch_size = trainer.config.horizon, trainer.config.batch_size
        assert horizon*batch_size == predicted_actions.shape[0]
        
        predicted_actions = predicted_actions.reshape(batch_size, horizon, -1)
        action = predicted_actions[t]
        obs, reward, done, _ = env.step(action)
        obs_history = np.append(obs_history, obs[None])
        assert obs_history[0].shape == obs[None].shape
        
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

    diffusion_model = GaussianDiffusion(unet, args.dsteps)
    action_model = InvKinematicsModel(obs_dim=obs_dim, action_dim=act_dim)
    models = {'diffusion': diffusion_model, 'inv_kinematics': action_model}
    ###
    #Train model
    ###
    train_config = TrainerConfig(train_num_steps=500000, train_inverse_kinematics=args.inv_kin, train_noise_prediction=not args.inv_kin, horizon=args.H)
    trainer = Trainer(models, dataset, train_config)
    if args.train:
        trainer.train()
    else:
        eval(trainer, "checkpoints/model-step-500000.pt", "checkpoints/ema-step-500001.pt", "checkpoints/inv_kin-step-205001.pt", device=DEVICE, horizon=args.H)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128, help='Horizon length') #refers to both training horizon and generation horizon
    parser.add_argument('--dsteps', type=int, default=128, help='Num diffusion steps')
    parser.add_argument('--train', action="store_true", help="Add flag to train or eval")
    parser.add_argument('--inv_kin', action="store_true", help="Add flag to train inv_kinematics instead of diffusion")
    args = parser.parse_args()
    main(args)

