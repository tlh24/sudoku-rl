import re 
import numpy as np
from pathlib import Path
from itertools import chain 
import sys 
import os 
from transformers import GPT2TokenizerFast 
from datasets import load_dataset 
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(home_dir)
from data.satnet_sudoku_data import Sudoku_SATNet
from torch.utils.data import TensorDataset
import torch 
import pickle 
import pandas as pd 
import csv 



from torch.utils.data import DataLoader, Dataset

def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string



def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data



class LargerSatNet:
    '''
    Returns solved puzzles as a 1d array. Each element in the array is digit in [0-8], corresponding to
    digits in [1,9]
    '''
    def __init__(self):
        with open(os.path.join(home_dir, 'data', 'easy_130k_solved.p'), 'rb') as file:
            self.data = pickle.load(file)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].flatten() - 1 #subtract 1 so we are zero indexed, i.e elemenets in [0,8] 
        item = item.astype(int)
        return item

def read_rrn_csv(file_path):
    '''
    Given rrn file, returns boards, solutions where boards is numpy array (num_puzzles, 81) with empty cells and solutions
    is (num_puzzles, 81) with no empty cells. Each sequence has values in [0,8] for filled digits and empty cells -100
    '''
    print("Reading %s..." % file_path)
    with open(file_path) as f:
        initial_puzzles, solutions = [], [] 
        reader = csv.reader(f, delimiter=',')
        for q,a in reader:
            # convert to satnet format where empty cells have -100 and values in [0,8]
            initial_board_digits = list(q)
            initial_board_digits = [int(digit_char)-1 if digit_char != "0" else -100 for digit_char in initial_board_digits]
            initial_puzzles.append(initial_board_digits)
            
            solution_digits = list(map(int, list(a)))
            # convert to zero index
            solution_digits = [digit-1 for digit in solution_digits]
            solutions.append(solution_digits)
        
        return np.stack(initial_puzzles), np.stack(solutions)

class TensDataset:
    def __init__(self, tensor: torch.Tensor):
        # tensor is shape (num_samples, 81)
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]
    
    def __getitem__(self, idx):
        return self.tensor[idx] 

        
        

    
def get_dataset(dataset_path: str, mode, with_initial_puzzles=False, cache_dir=None, block_size=1024, num_proc=8):
    '''
    Loads numpy dataset and returns a pytorch dataset
    dataset_path: (str) Should be filepath of numpy file that is of shape [num_samples, sample_length]
    '''
    if dataset_path == 'wikitext2':
        dataset = load_dataset("wikitext", name='wikitext-2-raw-v1', cache_dir=cache_dir)
        data = dataset[mode]
        
        detokenizer = wt_detokenizer 

        def _apply_tokenizer(detokenizer):
            def detok(text):
                for i,t in enumerate(text, 0):
                    text[i] = detokenizer(t)
                return text 
            return detok 
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        EOS = tokenizer.encode(tokenizer.eos_token)[0]

        def preprocess_and_tokenize(example):
            text = example["text"]
            if detokenizer is not None:
                text = _apply_tokenizer(detokenizer)(text)
            
            tokens = tokenizer(text, return_attention_mask=False)
            for token in tokens['input_ids']:
                token.append(EOS)
            return tokens 
        
        tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
        tokenized_dataset = tokenized_dataset.remove_columns("text")

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
        chunked_dataset = chunked_dataset.with_format('torch')
        return chunked_dataset

    elif dataset_path == 'satnet':
            dataset = Sudoku_SATNet()
            indices = list(range(len(dataset)))
            test_dataset = torch.utils.data.Subset(dataset, indices[-1000:])
            val_dataset = torch.utils.data.Subset(dataset, indices[8000:-1000])
            train_dataset = torch.utils.data.Subset(dataset, indices[:8000])
            if mode == 'train':
                return train_dataset
            elif mode == 'validation':
                return val_dataset
            else:
                return test_dataset
    elif dataset_path == 'larger_satnet':
        dataset = LargerSatNet()
        indices = list(range(len(dataset)))
        test_dataset = torch.utils.data.Subset(dataset, indices[int(0.9*len(dataset)):])
        val_dataset = torch.utils.data.Subset(dataset, indices[int(0.8*len(dataset)):int(0.9*len(dataset))])
        train_dataset = torch.utils.data.Subset(dataset, indices[:int(0.8*len(dataset))])
     
        if mode == 'train':
            return train_dataset
        elif mode == 'validation':
            return val_dataset
        else:
            return test_dataset    
    elif dataset_path == 'rrn':
        if mode == "train":
            boards, solutions = read_rrn_csv(os.path.join(home_dir, 'data', 'sudoku-hard', 'train.csv'))
        elif mode == "validation":
            boards, solutions = read_rrn_csv(os.path.join(home_dir, 'data', 'sudoku-hard', 'valid.csv'))
        elif mode == "test":
            boards, solutions = read_rrn_csv(os.path.join(home_dir, 'data', 'sudoku-hard', 'test.csv'))
        else:
            raise ValueError()

        boards_tens = torch.from_numpy(boards)
        solutions_tens = torch.from_numpy(solutions)
        if with_initial_puzzles:
            board_ds = TensDataset(boards_tens)
            solution_ds = TensDataset(solutions_tens)
            return board_ds, solution_ds
        
        solution_ds = TensDataset(solutions_tens)
        return solution_ds
    else:
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

    train_set = get_dataset(config['data']['train'], "train")
    valid_set = get_dataset(config['data']['valid'], "validation")

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

