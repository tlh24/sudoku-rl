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
import glob 

def reformat(seq):
    if len(seq.split(" ")) <= 1:
        seq = seq.replace("[AbLC]", "").replace("[AbHC]", "").replace("[Ag]", "")
        return seq 
    else:
        return seq


class LabeledDataset(Dataset):
    '''
    Creates a labeled dataset in the form of Protein Design with Guided Discrete Diffusion
    
    sudoku_dataset: pytorch dataset which returns puzzle sequence tensor of shape (81,) with elements with values in [0,8]
        corresponding to digits [1,9]
    '''
    def __init__(self, config, split):
        super().__init__()
        self.dataset_name = os.path.basename(config.data_dir)
        data_fn = os.path.join(config.data_dir, split)
        df = pd.concat([pd.read_csv(fn) for fn in glob.glob(data_fn)])
        tokenizer = transformers.BertTokenizerFast(
                vocab_file=config.vocab_file, 
                do_lower_case=False,
        )
        self.inputs = []
        assert config.use_alignment_tokens 

        for seq in tqdm.tqdm(df["full_seq"].values, total=len(df)):
            seq = reformat(seq)
            seq = tokenizer.convert_tokens_to_ids(seq.split(" "))      
            seq = torch.Tensor(seq).int() 
            list_token_ids = seq.tolist()
            if len(seq) != 300:
                continue 
            seq = seq[:81]
            corrupt_mask = torch.ones_like(seq)
            attn_mask = torch.ones_like(seq)
            self.inputs.append({
                "seq": seq,
                "corrupt_mask": corrupt_mask,
                "attn_mask": attn_mask,
            })
        
        assert config.target_cols is None 
        self.inputs = self.inputs[:config.max_samples]


        

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        vals = self.inputs[index]

        retval = {
            "attn_mask": vals["attn_mask"],
            "corrupt_mask": vals["corrupt_mask"],
            "seq": vals["seq"].long(),
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

def get_loaders(config):
    dsets = [LabeledDataset(config, split) for split in [config.train_fn, config.val_fn]]

    effective_batch_size = config.batch_size
    if torch.cuda.is_available():
        effective_batch_size = int(config.batch_size / torch.cuda.device_count())

    assert effective_batch_size == config.batch_size 
    loaders = [
        DataLoader(
            dataset=ds,
            batch_size=effective_batch_size,
            shuffle=(i == 0),
            num_workers=config.loader_workers,
            pin_memory=True,
        )
        for i, ds in enumerate(dsets)
    ]

    return loaders