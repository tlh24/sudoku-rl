import numpy as np 
import torch
import pdb
import sys
import os  
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname('./..'))
from utils import Trainer, TrainerConfig 
from model import TemporalUnet, GaussianDiffusion, InvKinematicsModel
from datasets.data import LimitsNormalizer

class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, horizon):
        #self.m = torch.randn(dataset_size) 
        #self.b = torch.randn(dataset_size)
        self.m = torch.randint(0, 2, (dataset_size,)) * 2 - 1
        self.b = torch.zeros(dataset_size)

        x = torch.linspace(-1,1, horizon) #(horizon,)
        y = torch.outer(self.m, x) + \
            torch.outer(self.b, torch.ones_like(x)) #(dataset_size, horizon)
        x = x.unsqueeze(0).expand(dataset_size, -1) 
        assert x.shape == y.shape 

        data = torch.stack([x,y,x,y], -1)
        max = torch.max(data)
        min = torch.min(data)
        normalized_data = (data - min) / (max - min)
        self.data = normalized_data*2 - 1
        

    def plotIt(self): 
        for j in range(1000): 
            i = np.random.randint(self.data.shape[0])
            plt.plot(self.data[i, :, 0], self.data[i, :, 1])
        plt.savefig('test.png')
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].numpy(), np.zeros(2)
    
class FuzzyLinearDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, horizon=64):
        x = np.linspace(-1,1, horizon)[None] 
        x = np.repeat(x, dataset_size, axis=0) #(batch_size, horizon)
        y = x + np.random.randn(*x.shape)*0.1   
        dataset = np.stack([x,y,x,y], axis=-1) #(batch_size, horizon, 4)
        unrolled_dataset = dataset.reshape(-1,4) #(batch_size*horizon, 4)   
        self.normalizers = self.get_normalizers(unrolled_dataset)
        norm_unrolled_dataset = self.normalizers['observations'].normalize(unrolled_dataset)
        self.norm_dataset = norm_unrolled_dataset.reshape(dataset_size, horizon, 4) 
    def get_normalizers(self, dataset):
        '''
        Returns a dictionary where key is 'observations' and value is LimitsNormalizer
        '''
        normalizers = {}
        for key in ['observations']:
            normalizers[key] = LimitsNormalizer(dataset)
        return normalizers

    def plotIt(self): 
        for plan_idx in range(0, len(self.norm_dataset)):
            plt.plot(self.norm_dataset[plan_idx, :, 0], self.norm_dataset[plan_idx, :, 1])
        plt.savefig('plot_fuzzy_linear.png')
        
    def __len__(self):
        return self.norm_dataset.shape[0]

    def __getitem__(self, idx):
        return self.norm_dataset[idx], np.zeros(2)

def generate_plans():
    '''
    Returns plans of shape (num_envs, horizon, trans_dim)
    '''
    train_config = TrainerConfig(train_num_steps=500000, train_inverse_kinematics=False, train_noise_prediction=True, horizon=128)
    obs_dim, act_dim = 4, 2 #hard-coded for maze2d
    unet = TemporalUnet(horizon=128, cond_dim = None, transition_dim=obs_dim, dim = 128, dim_mults=(1,2,4,8))

    diffusion_model = GaussianDiffusion(unet, 128)
    action_model = InvKinematicsModel(obs_dim=4, action_dim=2)
    models = {'diffusion': diffusion_model, 'inv_kinematics': action_model}
    dataset = LinearDataset(2000, 128)
    trainer = Trainer(models, dataset, train_config, save_folder="./test_checkpoints")
    trainer.load("test_checkpoints/diffusion-step-3001.pt", None, None)

    shape = (32, 128, 4)
    sample = trainer.models['diffusion'].p_sample_loop(shape, {})
    return sample 

def visualize_plan(plan):
    '''
    plan: (horizon, trans_dim)
    '''
    x = plan[:, 0].cpu().numpy()
    y = plan[:, 1].cpu().numpy()
    plt.plot(x,y)
    plt.savefig('test_generated_plan.png')


if __name__ == '__main__':
    breakpoint()
    d = FuzzyLinearDataset(1000, 64)
    d.plotIt()
    #plans = generate_plans()
    #visualize_plan(plans[0])
