import torch 
from torch import nn, einsum
import torch.nn.functional as F
import pdb
from torch.utils import data
from functools import partial 
import numpy as np
import math
import einops
from einops.layers.torch import Rearrange

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    
    def forward(self,x):
        return self.conv(x)
    
class Conv1dBlock(nn.Module):
    '''
    Conv1d -> GroupNorm -> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish()
        )
    
    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):
    '''
    residual temporal block from Decision Diffuser
    '''

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        '''
        embed_dim: (int) dimension size of the time embedding
        '''
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        #TODO: add attention as in replanner if not working
        return out + self.residual_conv(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TemporalUnet(nn.Module):
    def __init__(self, horizon, transition_dim, cond_dim=None, dim=128, dim_mults=(1,2,4,8), kernel_size=5):
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim*m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"Temporal Unet in/out dimensions: {in_out}")

        self.cond_dim = cond_dim 
        
        # define our time embedding network
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim),
                                      nn.Linear(dim, dim*2),
                                      nn.Mish(),
                                      nn.Linear(dim*2, dim))

        # define our conditional network TODO: change if doesn't work
        if self.cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                    nn.Linear(cond_dim, dim * 2),
                    nn.Mish(),
                    nn.Linear(dim * 2, dim))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # double embed_dim (time embed dim) if add conditional info, see forward() 
        if cond_dim is not None and cond_dim > 0:
            embed_dim = 2*dim 
        else:
            embed_dim = dim 
        
        # Downsampling code
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(
                    dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size
                ),
                ResidualTemporalBlock(
                    dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            if not is_last:
                horizon = horizon//2
            
        # Middle code
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon)
        
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon)

        # Upsampling code
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out*2, dim_in, embed_dim=embed_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
            if not is_last:
                horizon = horizon*2
        
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1)
        )

    def forward(self, x, cond, time):
        '''
        x: [batch x horizon x transition]
        returns: [batch x horizon]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)

        if self.cond_dim is not None and self.cond_dim > 0:
            # TODO: if doesn't work, add the decision diffuser masked bernoulli conditional code
            h = self.cond_mlp(cond)
            t = torch.cat([h,t], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x,t)
            x = resnet2(x,t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x,t)
        x = self.mid_block2(x,t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x,h.pop()), dim=1)
            x = resnet(x,t)
            x = resnet2(x,t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x   

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
        x = np.linspace(0,steps,steps)
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
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        return self.p_losses(obs, t, act)
