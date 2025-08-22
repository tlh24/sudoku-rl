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

def reformat(seq: str):
    if len(seq.split(" ")) <= 1:
        seq = seq.replace("[AbLC]", "").replace("[AbHC]", "").replace("[Ag]", "")
        return seq 
    else:
        return seq
    
vocab_blocks = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "-"]
reduced_vocab_blocks = vocab_blocks[:9]

def sequence_to_blocks(seq: str):
    # converts sequence string to a list of blocks that represent amino acids
    blocks = seq.split(" ")
    amino_blocks = list(filter(lambda x: x in vocab_blocks, blocks))
    return amino_blocks

def reduced_process_seq(seq: str):
    # converts sequence string to a list of string digits.
    blocks = seq.split(" ")
    amino_blocks = list(filter(lambda x: x in reduced_vocab_blocks, blocks))
    # convert to digits
    mapping = {elm: str(reduced_vocab_blocks.index(elm)) for elm in reduced_vocab_blocks}

    digit_sequence = list(map(lambda x: mapping[x], amino_blocks))
    return digit_sequence
    #return amino_blocks

class SudokuLabeledDataset(Dataset):
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
            vocab_file=config.vocab_file, 
            do_lower_case=False,
        )
        self.inputs = []

        for i in range(0, len(self.sudoku_dataset)):
            board_seq = self.sudoku_dataset[i].tolist()
            board_seq_chars = list(map(str, board_seq))
            assert len(board_seq_chars) == 81 and isinstance(board_seq_chars, list)
            seq = self.tokenizer.convert_tokens_to_ids(board_seq_chars)
            seq = torch.Tensor(seq).int()
            corrupt_mask = torch.ones_like(seq)
            attn_mask = torch.ones_like(seq)
            self.inputs.append({
                "seq": seq,
                "corrupt_mask": corrupt_mask,
                "attn_mask": attn_mask
            })
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

class ProteinLabeledDataset(Dataset):
    '''
    Creates a labeled dataset in the form of Protein Design with Guided Discrete Diffusion
        Loads in from protein files and creates a dataset
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
            blocks = reduced_process_seq(seq)
            seq = tokenizer.convert_tokens_to_ids(blocks)      
            seq = torch.Tensor(seq).int() 

            if len(seq) < 81:
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

    labeled_dataset = SudokuLabeledDataset(dataset, config)

    dl = DataLoader(dataset = labeled_dataset, batch_size=config['batch_size'],\
                     shuffle=mode=='train',num_workers=config.loader_workers, pin_memory=True)
    return dl 

def get_protein_loaders(config):
    dsets = [ProteinLabeledDataset(config, split) for split in [config.train_fn, config.val_fn]]

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