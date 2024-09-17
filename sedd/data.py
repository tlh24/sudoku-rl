import re 
import numpy as np
from pathlib import Path
from itertools import chain 
import sys 
from transformers import GPT2TokenizerFast 
from datasets import load_dataset 

sys.path.append(Path(__file__).resolve().parent.parent) #add root dir path

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


def get_dataset(dataset_path: str, mode, cache_dir=None, block_size=1024, num_proc=8):
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

    train_set = get_dataset(config['data']['train'], "train", )
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

