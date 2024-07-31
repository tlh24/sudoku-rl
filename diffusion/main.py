from utils import set_seed, Trainer, TrainerConfig 
import argparse
import torch 
from torch.utils.data import DataLoader
import gym 
from datasets.data import SequenceDataset 
from model import TemporalUnet, GaussianDiffusion, InvKinematicsModel
import pdb 
import numpy as np 
import os 
from datetime import datetime
from tqdm import tqdm 
from constants import maze2d_medium_v1_max_episode_steps
from viz import show_diffusion, MuJoCoRenderer, viz_plan_over_time
from datasets.test_diffusion import LinearDataset
ENV_NAME = 'maze2d-medium-v1'
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

def get_action(plan, t: int):
    '''
    plan: sequence of states from t = 0 to t_final
    
    Returns the action at time t by calculating the difference in the x,y location 
    of plan[t+1] - plan[t]
    '''
    #TODO: need to check the shape to make sure that you index the shit right
    plan = torch.flatten(plan)
    return plan[t+1][:2] - plan[t][:2]

        
class EvalExperimenter:
    '''
    Evaluates the diffusion model and (optionally) visualizes the generated plans 
    '''
    def __init__(self, log_folder: str, num_experiments: int, num_envs: int, trainer, horizon: int,\
                 diffusion_loadpath: str, ema_loadpath: str, inv_kin_loadpath: str, max_episode_steps: int,\
                    is_random_agent: bool, viz_plans: bool):
        '''
        horizon: (int) Lnegth of the plan that gets generated 
        diffusion_loadpath: (str) Path to the diffusion model weights 
        ema_loadpath: (str) Path to ema diffusion model weights
        inv_kin_loadpath: (str) Path to inverse kinematics (action predictor) model weights 
        '''
        self.log_folder = log_folder
        self.num_experiments = num_experiments
        self.num_envs = num_envs 
        self.trainer = trainer 
        self.trainer.load(diffusion_loadpath, ema_loadpath, inv_kin_loadpath)
        self.horizon = horizon 
        self.max_episode_steps = max_episode_steps
        self.is_random_agent = is_random_agent
        self.viz_plans = viz_plans
        # experiment reward log file saved by file pointer fp 
        time_str = datetime.now().strftime('%Y-%m-%d-%H:%M')
        file_name = f"{'random_' if self.is_random_agent else ''}{self.num_experiments}runs_{self.num_envs}envs_eval_log_{time_str}.txt"
        self.file_path = os.path.join(self.log_folder, file_name)
        self.fp = open(self.file_path, 'w')


    def run_experiments(self):
        '''
        Runs self.eval() num_experiments many times 
        '''
        for exp_num in tqdm(range(1, self.num_experiments + 1)):
            self.eval(exp_num)
        
    def eval(self, exp_num: int, history_len=20):
        '''
        Runs one experiment and evaluates the diffusion model on num_envs many environments. 
            Runs until all envs terminate; then returns the mean reward across the envs, also optionally visualizes generated 
            diffusion plans  

        history_len: (int) This is the max number of states to condition the plan on.
            For HL=20, up to the first 20 states of the diffusion plan are the last (up to) 20 visited states
            In janner, they condition only on the last visited state, i.e HL=1
        '''
        # Load models and set to eval mode
        for name, model in self.trainer.models.items():
            model.eval()
        
        env_list = [gym.make(ENV_NAME) for _ in range(self.num_envs)]
        dones = [0 for _ in range(self.num_envs)]
        episode_rewards = [0 for _ in range(self.num_envs)]
        # shape is (num_envs, num_timesteps, obs_dim), where obs_dim is the dimension of the diffusion plan object for a fixed timestep 
        obs_history = np.array([env.reset()[None] for env in env_list]) 
        t = 0
        num_episode_steps = 0
        diffusion_plans = [] # (num_episode_steps x num_envs x horizon x obs_dim)

        # Iterate until all envs finish but cap at max episode steps taken
        while sum(dones) < self.num_envs and num_episode_steps < self.max_episode_steps:
            if not self.is_random_agent:
                norm_history = self.trainer.train_dataset.normalizers['observations'].normalize(obs_history[:, -history_len:]) 
                conditions = {} # key is time step t to replace, value is state at timestep t  
                for i in range(0, norm_history.shape[1]):
                    conditions[i] = to_torch(norm_history[:, i], dtype=DTYPE, device=DEVICE)
                
                shape = [self.num_envs, 128, 4]
                sample = self.trainer.models['diffusion'].p_sample_loop(shape, {})
                #sample = self.trainer.models['diffusion'].conditional_sample(conditions, self.horizon)
                num_envs, horizon = sample.shape[0], sample.shape[1]

                # save diffusion plans for viz
                if self.viz_plans:
                    unnormalized_sample = self.trainer.train_dataset.normalizers['observations'].unnormalize(sample.detach().cpu().numpy())
                    diffusion_plans.append(unnormalized_sample)

                comb_obs = self.trainer.models['inv_kinematics'].get_combined_obs(sample) #(num_envs*H, 2*obs_dim)
                predicted_actions = self.trainer.models['inv_kinematics'](comb_obs) #(num_envs* H-1, act_dim)
                assert (horizon-1)*self.num_envs == predicted_actions.shape[0]
                
                predicted_actions = predicted_actions.reshape(self.num_envs, horizon-1, -1)
                envs_action = predicted_actions[:, t]
            else:
                envs_action = [torch.from_numpy(env.action_space.sample()) for env in env_list]
            
            obs_arr = [] # contains the observations across envs for this timestep 
            for i in range(0, self.num_envs):
                obs, reward, done, _ = env_list[i].step(envs_action[i].detach().cpu().numpy())
                obs_arr.append(obs[None])
                if dones[i] == 1:
                    pass 
                elif done:
                    dones[i] = 1
                    episode_rewards[i] += reward 
                    print(f"Episode {i} : reward {episode_rewards[i]}")
                    self.fp.write(f"exp_{exp_num}_eps_{i}:{episode_rewards[i]}\n")
                else:
                    episode_rewards[i] += reward 
            
            obs_arr = np.stack(obs_arr, axis=0) # (num_envs, 1, obs_dim)

            obs_history = np.concatenate((obs_history, obs_arr), axis=1)
            # note that we condition on the last HL observations in our plan;
            # thus the action index should be at most HL-1 (try it for HL=1)
            t = min(history_len-1, t+1) 
            num_episode_steps += 1
        
        # log all timeout'ed episodes as the sum reward received within alloted time
        for i in range(0, self.num_envs):
            if dones[i] != 1:
                print(f"Timeout Episode {i} : reward {episode_rewards[i]}\n")
                self.fp.write(f"timeout_exp_{exp_num}_eps_{i}:{episode_rewards[i]}\n")
        
        # optionally visualize the diffusion plans and save them to a folder 
        if len(diffusion_plans) and self.viz_plans:
            renderer = MuJoCoRenderer(gym.make(ENV_NAME))
            diffusion_plans = np.stack(diffusion_plans, axis=0)
            breakpoint()
            
            first_env_plans = diffusion_plans[:,0,:,:]
            viz_plan_over_time(renderer, first_env_plans,\
                                 savefolder=f'images/exp_num{exp_num}')


def main(args=None):
    print(f"Arguments {vars(args)}")

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
        experimenter = EvalExperimenter("logging/", args.numexps, args.numenvs, trainer, args.H,\
                                         "checkpoints/diffusion-step-65001.pt", "checkpoints/ema-step-65001.pt",\
                                            "checkpoints/inv_kin-step-205001.pt", maze2d_medium_v1_max_episode_steps,\
                                                is_random_agent=False, viz_plans=True)
        experimenter.run_experiments()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128, help='Horizon length') #refers to both training horizon and generation horizon
    parser.add_argument('--numexps', type=int, default=3, help='Number of experiments to run') 
    parser.add_argument('--numenvs', type=int, default=20, help='Number of envs; the batch size of diffusion plan generation') 
    parser.add_argument('--dsteps', type=int, default=128, help='Num diffusion steps')
    parser.add_argument('--train', action="store_true", help="Add flag to train or eval")
    parser.add_argument('--inv_kin', action="store_true", help="Add flag to train inv_kinematics instead of diffusion")
    args = parser.parse_args()
    main(args)

