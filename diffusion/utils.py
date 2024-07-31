import random
import numpy as np
import torch
import math 
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math 
from tqdm import tqdm 
import os 
from copy import deepcopy
import pdb
import matplotlib.pyplot as plt

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta 
    
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    
    def update_average(self, old, new):
        if old is None:
            return new 
        return old * self.beta + (1 - self.beta)*new 

    

class TrainerConfig:
    batch_size = 32
    horizon=128
    learning_rate = 2e-5
    train_num_steps = 1000000
    print_loss_num_steps = 1000 #print training loss every _ steps
    save_model_num_steps = 10000
    patience = 5000 # num of steps to wait and see if loss decreases
    min_delta = 0.001 # how much loss needs to decrease by to update best_loss 
    ema_decay = 0.995
    step_start_ema=2000
    update_ema_every=10
    train_noise_prediction = False 
    train_inverse_kinematics = True


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    

class Trainer:
    def __init__(self, models, train_dataset, config, save_folder='./checkpoints'):
        '''
        models: (dict) Dictionary with models under keys 'diffusion' and 'inv_kinematics'
        train_dataset: (torch Dataset)  training trajectories
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {name: model.to(self.device) for name, model in models.items()}

        self.train_dataset = train_dataset
        pdb.set_trace()
        self.config = config
        self.train_dataloader = cycle(DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True))
        self.optimizers = {}
        for name, model in self.models.items():
            self.optimizers[name] = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        self.train_num_steps = config.train_num_steps
        self.print_loss_num_steps = config.print_loss_num_steps
        self.save_model_num_steps = config.save_model_num_steps
        self.save_folder=save_folder
        # TODO: if breaks, load previous ema model 
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0

        # ema for diffusion model 
        self.ema = EMA(config.ema_decay)
        self.ema_model = deepcopy(self.models['diffusion'])
        self.update_ema_every = config.update_ema_every 
        self.step_start_ema = config.step_start_ema 
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.models['diffusion'].state_dict()) 
    
    def step_ema(self):
        '''
        Update the ema for diffusion model 
        '''
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return 
        self.ema.update_model_average(self.ema_model, self.models['diffusion'])
    
    def train(self):
        fd_losslog = open('losslog.txt', 'w')
        if self.config.train_noise_prediction:
            training_log = open(os.path.join(self.save_folder, 'diffusion_training_log.txt'), 'w')
        if self.config.train_inverse_kinematics:
            training_log = open(os.path.join(self.save_folder, 'inv_kin_training_log.txt'), 'w')

        for _ in tqdm(range(self.train_num_steps)):
            #TODO: possible change to obs, act, state
            obs, act = next(self.train_dataloader)
            obs = obs.float().to(self.device)
            act = act.float().to(self.device)
            
            total_loss = 0
            if self.config.train_noise_prediction:
                loss = self.models['diffusion'](obs)
                total_loss += loss
            
            if self.config.train_inverse_kinematics:
                loss = self.models['inv_kinematics'].compute_loss(obs, act)
                total_loss += loss 
            
            total_loss.backward()
            
            if self.config.train_noise_prediction:
                self.optimizers['diffusion'].step() 
                self.optimizers['diffusion'].zero_grad()

                # update diffusion ema
                if (self.step % self.update_ema_every == 0):
                    self.step_ema()

            if self.config.train_inverse_kinematics:
                self.optimizers['inv_kinematics'].step() 
                self.optimizers['inv_kinematics'].zero_grad()

            if (self.step % self.print_loss_num_steps == 0):
                current_loss = total_loss.cpu().item()
                # print(f"Training loss at batch step {self.step+1}: {current_loss:.4f}")
                training_log.write(f"Batch {self.step+1}: {current_loss:.4f}\n")
                training_log.flush()

                # early stopping check
                if self.check_early_stopping(current_loss):
                    print(f"Stopping early at step {self.step+1}")
                    break

            if (self.step % self.save_model_num_steps == 0):
                print(f'Saving model at step {self.step+1}')
                self.save(self.step+1)
            
            self.step += 1
        self.save(self.step+1)
        print("Training completed.")
        training_log.close()
    
    def check_early_stopping(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += self.print_loss_num_steps
            if self.patience_counter >= self.patience:
                return True 
        return False 

    def save(self, step: int):
        '''
        Save the model into a folder
        ''' 
        if self.config.train_noise_prediction:
            torch.save(self.models['diffusion'].state_dict(), os.path.join(self.save_folder, f'diffusion-step-{step}.pt'))
            torch.save(self.ema_model.state_dict(), os.path.join(self.save_folder, f'ema-step-{step}.pt'))
        if self.config.train_inverse_kinematics:
            torch.save(self.models['inv_kinematics'].state_dict(), os.path.join(self.save_folder, f'inv_kin-step-{step}.pt'))

    def load(self, diffusion_load_path, ema_load_path, inv_kin_load_path):
        #TODO: if loading model to continue training, need to also load the last saved time step
        if diffusion_load_path:
            diffusion_state_dict = torch.load(diffusion_load_path)
            self.models['diffusion'].load_state_dict(diffusion_state_dict)
        if ema_load_path:
            ema_state_dict = torch.load(ema_load_path)
            self.ema_model.load_state_dict(ema_state_dict)
        if inv_kin_load_path:
            inv_kin_state_dict = torch.load(inv_kin_load_path)
            self.models['inv_kinematics'].load_state_dict(inv_kin_state_dict)
            

def set_seed(seed, env = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_cond(x, cond, trans_dim):
    '''
    cond: (dict) key is timestep t to replace, val is a state at time t
    trans_dim: (int) length of the state in the history of observations to condition on
    '''
    for t, val in cond.items():
        x[:, t, :trans_dim] = val.clone()
    return x
    
