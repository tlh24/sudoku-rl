import numpy as np
from pathlib import Path
import sys 
sys.path.append(Path(__file__).resolve().parent.parent) #add root dir path

from torch.utils.data import DataLoader, Dataset


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def get_dataset(dataset_path: str):
    '''
    Loads numpy dataset and returns a pytorch dataset
    dataset_path: (str) Should be filepath of numpy file that is of shape [num_samples, sample_length]
    '''

    class SequenceDataset(Dataset):
        def __init__(self, data):
            self.data = data 
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    sequences_arr = np.load(dataset_path) #(num_samples, seq_len)
    dataset = SequenceDataset(sequences_arr)
    return dataset


def get_dataloaders(config):
    if config['training']['batch_size'] % (config['ngpus'] * config['training']['accum']) != 0:
            raise ValueError(f"Train Batch Size {config['training']['batch_size']} is not divisible by {config['ngpus']} gpus with accumulation {config['training']['accum']}.")
    if config['eval']['batch_size'] % (config['ngpus'] * config['training']['accum']) != 0:
        raise ValueError(f"Eval Batch Size for {config['eval']['batch_size']} is not divisible by {config['ngpus']} gpus with accumulation {config['training']['accum']}.")

    train_set = get_dataset(config['data']['train'])
    valid_set = get_dataset(config['data']['valid'])

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config['training']['batch_size'] // (config['ngpus'] * config['training']['accum']),
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config['training']['batch_size'] // (config['ngpus'] * config['training']['accum']),
        num_workers=4,
        pin_memory=True,
        shuffle=True
    ))
    return train_loader, valid_loader

