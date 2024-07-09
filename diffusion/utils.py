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

def cycle(dl):
    while True:
        for data in dl:
            yield data

class TrainerConfig:
    batch_size = 32
    learning_rate = 2e-5
    train_num_steps = 100000
    print_loss_num_steps = 1000 #print training loss every _ steps
    save_model_num_steps = 5000

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    

class Trainer:
    def __init__(self, model, train_dataset, config, save_folder='./checkpoints'):
        '''
        train_dataset: (torch Dataset)  training trajectories
        '''
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.train_dataloader = cycle(DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.train_num_steps = config.train_num_steps
        self.print_loss_num_steps = config.print_loss_num_steps
        self.save_model_num_steps = config.save_model_num_steps
        self.save_folder=save_folder
        # TODO: if breaks, load previous ema model 
    
    def train(self):
        training_log = open(os.path.join(self.save_folder, 'training_log.txt'), 'w')

        for i in tqdm(range(self.train_num_steps)):
            #TODO: possible change to obs, act, state
            obs, act = next(self.train_dataloader)
            obs = obs.float().to(self.device)
            act = act.float().to(self.device)

            #TODO: add more losses if need be
            loss = self.model(obs, act)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            #TODO: add ema and saving checkpoints if need be
            if (i % self.print_loss_num_steps == 0):
                print(f"Training loss at batch step {i+1}: {loss.cpu().item():.4f}")
                training_log.write(f"Batch {i+1}: {loss.cpu().item():.4f}")

            if (i % self.save_model_num_steps == 0):
                print(f'Saving model at step {i+1}')
                self.save(i+1)
        self.save(i+1)
        print("Training completed.")
        training_log.close()

    def save(self, step: int):
        '''
        Save the model into a folder
        ''' 
        torch.save(self.model.state_dict(), os.path.join(self.save_folder, f'model-step-{step}.pt'))
    
    def load(self, load_path):
        model_state_dict = torch.load(load_path)
        self.model.load_state_dict(model_state_dict)


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
