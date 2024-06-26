import random
import numpy as np
import torch
import math 
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math 
from tqdm import tqdm 

def cycle(dl):
    while True:
        for data in dl:
            yield data

class TrainerConfig:
    batch_size = 32
    learning_rate = 2e-5
    train_num_steps = 100000
    print_loss_num_steps = 10 #print training loss every _ steps

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    

class Trainer:
    def __init__(self, model, train_dataset, config):
        '''
        train_dataset: (torch dataset)  training trajectories
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
        # TODO: if breaks, load previous ema model 
    
    def train(self):
        for i in tqdm(range(self.train_num_steps)):
            #TODO: possible change to obs, act, state
            obs, act = next(self.train_dataloader)
            obs = obs.float().to(self.device)
            act = act.float().to(self.device)

            #TODO: add more losses if need be
            loss = self.model(obs, act)
            loss.backwards()

            self.optimizer.step()
            self.optimizer.zero_grad()

            #TODO: add ema and saving checkpoints if need be
            if (i % self.print_loss_num_steps == 0):
                print(f"Training loss at batch step {i+1}: {loss.cpu().item():.4f}")

        print("Training completed.")




def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
