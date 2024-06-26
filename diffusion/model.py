import torch 
from torch import nn, einsum
import torch.nn.functional as F
import pdb
from torch.utils import data
from functools import partial 
import numpy as np


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, timesteps):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        ###
        # Diffusion Constants
        ###
        
        #TODO: change beta init if bad
        betas = GaussianDiffusion.cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1]) 

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    @staticmethod 
    def cosine_beta_schedule(timesteps, s=0.008):
        '''
        Defines beta as in https://openreview.net/forum?id=-NEXDKk8gZ
        '''
        steps = timesteps + 1
        x = np.linspace(0 ,steps, steps)
        alphas_cumprod = np.cos(( (x/steps)+s ) / (1+s)*np.pi*0.5)**2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, a_min=0, a_max=0.999)
        

    @staticmethod
    def extract(a, t, x_shape):
        '''
        Given a batch of sequences of elements {a_t}, extracts all of the 
        a_t elements corresponding to number t  
        '''
        
        b, *_ = t.shape 
        out = a.gather(-1 ,t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        '''
        t: (batch_size,) A random integer timestep cloned batch_size times

        Returns a time-t noised version of x_start 
        '''
        sample = (
            GaussianDiffusion.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
            GaussianDiffusion.extract(self.sqrt_one_minus_alphas_cumprod, t ,x_start.shape) * noise
        )
        return sample   

    def p_losses(self, obs, t, act=None):
        '''
        t: (batch_size,) A random integer timestep cloned batch_size times
        '''
        x_start = obs 
        device = x_start.device 
        noise = torch.randn(*x_start.size(), device=device)
        x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise)
        x_recon = self.denoise_fn(x_noisy, x_start, t)
        assert x_start.shape == x_recon.shape
        
        loss = F.mse_loss(noise, x_recon)
        return loss 
    

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        device = obs.device 
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        return self.p_losses(obs, t, act)
