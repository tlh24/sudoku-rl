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
from utils import apply_cond
from tqdm import tqdm

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

    def forward(self, x, time, cond):
        '''
        x: [batch x horizon x transition]
        time: [batch,]

        returns: [batch x horizon]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)

        if self.cond_dim is not None and self.cond_dim > 0:
            raise ValueError(f"should not have conditional info in diffusion training yet")
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
    def __init__(self, denoise_fn, timesteps, trans_dim=4):
        '''
        trans_dim: (int) The size of the vector corresponding to a particular timestep that is being diffused.
            For state only, it is the dim of the state vector
        ''' 
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        self.trans_dim = trans_dim
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

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = to_torch(betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


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

        t: (batch_size, )
        '''
        
        b, *_ = t.shape 
        out = a.gather(-1 ,t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            GaussianDiffusion.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)*x_t - 
            GaussianDiffusion.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)*noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            GaussianDiffusion.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            GaussianDiffusion.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = GaussianDiffusion.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = GaussianDiffusion.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        '''
        x: the noisy input at time t 
        t: (batch_size, ) torch.full of the int timestep t
        cond: dictionary with key as timestep to replace, value is obs 
        '''
        #TODO: add conditioning information to noise prediction 
        pred_noise = self.denoise_fn(x, t, cond=None)
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    
    @torch.no_grad()
    def p_sample(self, x, cond, timesteps):
        '''
        Return a one-step reverse diffused x
        
        x: the noisy input at time t
        timesteps: (batch_size, ) torch.full of the int timestep t 
        '''

        num_envs = x.shape[0]
        device = x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=timesteps)
        # TODO: can change to 0.1 as in replanner
        # no noise when t == 0 
        noise = 0.5*torch.randn_like(x) if timesteps[0] > 0 else 0 
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_img
        
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        '''
        Generates a plan based on the given condition dictionary

        cond: (dict) key is timestep to replace, value is timestep state
        shape: (num_envs, horizon, trans_dim)
        '''
        device = self.betas.device
        num_envs = shape[0]
        #TODO: if doesn't work delete the 0.5
        start_x = 0.5*torch.randn(shape, device=device)
        x = apply_cond(start_x, cond, self.trans_dim)

        for t in tqdm(reversed(range(0, self.timesteps))):
            timestep = torch.full((num_envs,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timestep)
            x = apply_cond(x, cond, self.trans_dim)

        return x

    def conditional_sample(self, cond, horizon: int):
        '''
        cond: (dict) key is timestep to replace and value is the obs at that timestep to condition on 
        horizon: (int) Length in timesteps of the plan to be generated
        '''
        device = self.betas.device
        num_envs = len(cond[0])
        sample_shape = (num_envs, horizon, self.trans_dim)
        return self.p_sample_loop(sample_shape, cond)


    def q_sample(self, x_start, t, noise=None):
        '''
        Returns a time-t noised version of x_start 
        
        x_start: (torch.Tensor) initial uncorrupted data
        t: (batch_size,) A random integer timestep cloned batch_size times
        '''
        sample = (
            GaussianDiffusion.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
            GaussianDiffusion.extract(self.sqrt_one_minus_alphas_cumprod, t ,x_start.shape) * noise
        )
        return sample   

    def p_losses(self, x_start, batch_t):
        '''
        x_start: (torch.Tensor) initial uncorrupted data
        batch_t: (batch_size,) A random integer timestep cloned batch_size times
        '''
        device = x_start.device 
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start,t=batch_t,noise=noise)
        #TODO: figure out why diffuser applies conditioning to the x_noisy 
        noise_hat = self.denoise_fn(x_noisy, batch_t, cond=None)
        assert noise.shape == noise_hat.shape
        
        loss = F.mse_loss(noise, noise_hat)
        return loss 
    
    def forward(self, obs):
        '''
        Runs a training iteration of diffusion noise prediction and returns a loss 

        obs: (torch.Tensor)
        '''

        batch_size = obs.shape[0]
        device = obs.device 
        batch_t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        return self.p_losses(obs, batch_t)

class InvKinematicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim 

        self.model = nn.Sequential(
            nn.Linear(2 * self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )
    
    def compute_loss(self, obs, action):
        '''
        obs: (batch, horizon, obs_dim)
        action: (batch, horizon, act_dim)
        '''
        x_t = obs[:, :-1]
        x_t_1 = obs[:, 1:]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2*self.obs_dim)
        a_t = action[:,:-1].reshape(-1, self.action_dim)

        predicted_action = self(x_comb_t)
        return F.mse_loss(predicted_action, a_t)

    def forward(self, x_comb):
        '''
        x_comb: (_, 2*obs_dim)
        '''
        return self.model(x_comb)

