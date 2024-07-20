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
NUM_ENVS = 10

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
    pdb.set_trace()
    env_list = [gym.make(ENV_NAME) for _ in range(NUM_ENVS)]
    dones = [0 for _ in range(NUM_ENVS)]
    episode_rewards = [0 for _ in range(NUM_ENVS)]

    obs_history = np.array([env.reset()[None] for env in env_list]) # shape is (num_envs, num_timesteps, state_dim)
    t = 0
    while sum(dones) < NUM_ENVS:
        norm_history = trainer.train_dataset.normalizers['observations'].normalize(obs_history[:, -history_len:]) 
        conditions = {} # key is time step t to replace, value is state at timestep t  
        for i in range(0, norm_history.shape[1]):
             conditions[i] = to_torch(norm_history[:, i], dtype=DTYPE, device=DEVICE)
        
        sample = trainer.models['diffusion'].conditional_sample(conditions, horizon)
        num_envs, horizon = sample.shape[0], sample.shape[1]
        # TODO: add generated plan visualization, allow for more than the first diffusion plan 
        if t == 0:
            unnormalized_sample = trainer.train_dataset.normalizers['observations'].unnormalize(sample.detach().cpu().numpy())
            # visualize_plan(unnormalized_sample)
        
        comb_obs = trainer.models['inv_kinematics'].get_combined_obs(sample) #(num_envs*H, 2*obs_dim)
        predicted_actions = trainer.models['inv_kinematics'](comb_obs) #(num_envs* H-1, act_dim)
        assert (horizon-1)*num_envs == predicted_actions.shape[0]
        
        predicted_actions = predicted_actions.reshape(num_envs, horizon-1, -1)
        envs_action = predicted_actions[:, t]
        obs_arr = []
        for i in range(0, num_envs):
            obs, reward, done, _ = env_list[i].step(envs_action[i].detach().cpu().numpy())
            obs_arr.append(obs[None])
            
            if dones[i] == 1:
                pass 
            elif done:
                dones[i] = 1
                episode_rewards[i] += reward 
                print(f"Episode {i} : reward {episode_rewards[i]}")
            else:
                episode_rewards[i] += reward 
        
        obs_arr = np.stack(obs_arr, axis=0)

        obs_history = np.concatenate((obs_history, obs_arr), axis=1)
        # note that we condition on the last HL observations in our plan;
        # thus the action index should be at most HL-1 (try it for HL=1)
        t = min(history_len-1, t+1) 
        

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
        eval(trainer, "checkpoints/diffusion-step-65001.pt", "checkpoints/ema-step-65001.pt", "checkpoints/inv_kin-step-205001.pt", device=DEVICE, horizon=args.H)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128, help='Horizon length') #refers to both training horizon and generation horizon
    parser.add_argument('--dsteps', type=int, default=128, help='Num diffusion steps')
    parser.add_argument('--train', action="store_true", help="Add flag to train or eval")
    parser.add_argument('--inv_kin', action="store_true", help="Add flag to train inv_kinematics instead of diffusion")
    args = parser.parse_args()
    main(args)

