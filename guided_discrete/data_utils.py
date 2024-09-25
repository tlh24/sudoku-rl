import os 

from sedd.data_utils import get_dataset
import os 
from pathlib import Path 
import tqdm 
import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader 
import transformers 


class LabeledDataset(Dataset):
    '''
    Creates a labeled dataset in the form of Protein Design with Guided Discrete Diffusion
    
    sudoku_dataset: pytorch dataset which returns puzzle sequence tensor of shape (81,) with elements with values in [0,8]
        corresponding to digits [1,9]
    '''
    def __init__(self, sudoku_dataset, config):
        super().__init__()
        self.sudoku_dataset = sudoku_dataset
        self.config = config 
        self.tokenizer = transformers.BertTokenizerFast(
            vocab_file=self.config.vocab_file,
            do_lower_case=False,
        )
        

    def __len__(self):
        return len(self.sudoku_dataset)
    
    def __getitem__(self, index):
        board_seq = self.sudoku_dataset[index] #either a tensor of 81 numbers or an array of 81 number
        if isinstance(board_seq, torch.Tensor):
            board_seq_list = board_seq.tolist()
        elif isinstance(board_seq, np.ndarray):
            board_seq_list = board_seq.tolist()
        
        board_seq_chars = list(map(str, board_seq_list))
        seq = self.tokenizer.convert_tokens_to_ids(board_seq_chars)
        seq = torch.Tensor(seq).int()
        
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

    labeled_dataset = LabeledDataset(dataset, config)

    dl = DataLoader(dataset = labeled_dataset, batch_size=config['batch_size'], shuffle=mode=='train', pin_memory=True)
    return dl 