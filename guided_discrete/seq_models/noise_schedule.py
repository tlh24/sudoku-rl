import math
import copy 
import numpy as np 
import torch 
import torch.nn as nn 
import transformers 

class DiscreteCorruptionSchedule:
    def __init__(
            self,
            vocab_file,
            timesteps=1000,
            noise_schedule='linear',
            noise_type='mask'
    ):
        self.timesteps = timesteps

        tokenizer = transformers.BertTokenizerFast(
            vocab_file=vocab_file,
            do_lower_case=False 
        )

        if noise_type == "mask":
            self.noise_token_ids = tokenizer.mask_token_id * torch.ones(1, dtype=torch.long)
        else:
            raise NotImplementedError()
        
        if noise_schedule == 'linear':
            self.mask_rates = np.linspace(
                0,1,timesteps,dtype=np.float64
            )
        elif noise_schedule == 'cosine':
            self.mask_rates = 1 - (np.cos(
                np.pi * np.linspace(0,1,timesteps, dtype=np.float64)
            ) + 1.0) / 2 
    
    def sample_prior(self, shape, device):
        '''
        Returns tensor filled with tokenizer.mask_token_id of shape "shape"
        '''
        noise_id_idxs = torch.randint(low=0, high=len(self.noise_token_ids), size=shape, device=device)
        corrupt_ids = torch.take(self.noise_token_ids.to(noise_id_idxs), noise_id_idxs)
        return corrupt_ids

    def corrupt(self, input_ids, timesteps, corrupt_mask=None):
        '''
        Returns a corrupted sequences containing some [MASK] tokens and also the boolean mask tensor
            which indicates which elements to corrupt to [MASK]

        input_ids: represents the data sequences, of shape (batch_size, seq len)
        timesteps: (batch_size,) a vector that contains the constant t that represents the timestep 
        '''
        mask_nums = torch.rand_like(input_ids, dtype=torch.float32)
        # mask is a vector of booleans of shape input_ids where each element is True with probability
        # mask_rates[t] (think \Beta_t)
        mask = torch.zeros_like(mask_nums, dtype=torch.bool)
        for i,t in enumerate(timesteps):
            mask[i] = mask_nums[i] < self.mask_rates[t] 
        
        if corrupt_mask is not None:
            mask = (mask * corrupt_mask).bool()

        noise_token_ids = self.noise_token_ids.to(input_ids.device)

        new_ids = copy.deepcopy(input_ids)
        for i,t in enumerate(timesteps):
            # replace all the input ids with the mask id where the mask boolean is True  

            noise_id_idxs = torch.randint_like(input_ids[i], 0, len(self.noise_token_ids))
            # corruptids is a tensor of mask_id of shape input_ids  
            corrupt_ids = torch.take_along_dim(noise_token_ids, noise_id_idxs.long(), dim=0)
            new_ids[i] = torch.where(mask[i], corrupt_ids, new_ids[i])
        
        return new_ids, mask 