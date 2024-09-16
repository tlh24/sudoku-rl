from ..sedd.data_utils import get_dataset
import os 
from pathlib import Path 
import tqdm 
import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import Dataset, Dataloader 




class LabeledDataset(Dataset):
    '''
    Creates a labeled dataset in the form of Protein Design with Guided Discrete Diffusion
    
    sudoku_dataset: pytorch dataset which returns puzzle sequence tensor of shape (81,)
    '''
    def __init__(self, sudoku_dataset):
        super().__init__()
        self.sudoku_dataset = sudoku_dataset

    def __len__(self):
        return len(self.sudoku_dataset)
    
    def __getitem__(self, index):
        seq = self.sudoku_dataset[index]
        #TODO: confirm that attention and corrupt masks are all ones
        retval = {
            "attn_mask": torch.ones_like(seq),
            "corrupt_mask": torch.ones_like(seq),
            "seq": seq 
        }
        return retval

    

def get_dataloader(config, mode: str):
    '''
    Returns sudoku dataloaders which only contain necessary corrupt masks and 
        attention masks
    
    mode: (str) String either in 'train', 'validation' or 'test'
    '''
    if mode == 'train':
        dataset = get_dataset(config.train, 'train')
    elif mode == "validation":
        dataset = get_dataset(config.valid, 'validation')
    else:
        dataset = get_dataset(config.valid, 'test')

    dl = Dataloader(dataset = dataset, batch_size=config['batch_size'], shuffle=mode=='train', pin_memory=True)
    return dl 