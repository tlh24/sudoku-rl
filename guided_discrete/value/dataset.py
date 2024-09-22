import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)
from sedd.data_utils import get_dataset
import torch.nn.functional as F 
import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np 

def noise_one_hot(one_hot_tensor, eps=0.05):
    '''
    Implements typical label smoothing: with K classes, the value of 1 becomes 1-eps 
        and the value of 0's become eps/(k-1)
    Returns a new tensor with smoothed labels
    '''
    K = one_hot_tensor.shape[-1]
    smoothed = torch.full_like(one_hot_tensor, fill_value=eps/(K-1), dtype=torch.float)
    smoothed.masked_fill_(one_hot_tensor != 0, 1-eps)
    #assert int(torch.sum(smoothed).item()) == len(smoothed)

    return smoothed 

class OneHotNoisy(Dataset):
    '''
    One hot encodes sequence of board digits. Also (optionally) adds label smoothing. 
        Eps is uniformly sampled in range [eps_low, eps_high]
    '''
    def __init__(self, tens_dataset, is_noisy, eps_low, eps_high):
        self.tens_dataset = tens_dataset
        self.is_noisy = is_noisy
        self.eps_low = eps_low 
        self.eps_high = eps_high
         
    def __len__(self):
        return len(self.tens_dataset)
    
    def __getitem__(self, idx):
        board_tens = self.tens_dataset[idx] #sequence of ints 0-8 corresponding to values 1-9
        one_hot_tens = F.one_hot(board_tens, num_classes=9)
        if self.is_noisy:
            eps = np.random.uniform(self.eps_low, self.eps_high)
            smoothed_tens = noise_one_hot(one_hot_tens, eps)
            return smoothed_tens
        return one_hot_tens
        

def get_one_hot_dataset(dataset_name, is_noisy, eps_low, eps_high):
    '''
    Returns a (noisy) one_hot encoding of the board digits
        
    eps_low: (float) The lowest eps to be used in label smoothing 
    eps_high: (float) Highest eps to be used in label smoothing 
    '''
    tens_dataset = get_dataset(dataset_name, "train")
    return OneHotNoisy(tens_dataset, is_noisy, eps_low, eps_high)  


if __name__ == "__main__":
    ds = get_one_hot_dataset("rrn", True, 0.01, 0.1)
    first = ds[0]
    breakpoint()
    print("Completed") 