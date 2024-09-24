import os 
import sys 
import torch 
home_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(home_dir)
import sedd.data_utils as data 
import numpy as np 
from torch.utils.data import Subset

import pickle 
from torch.utils.data import Subset
from sedd.utils import isValidSudoku

class LargerSatNetInitial:
    '''
    Returns initial puzzles as a 1d array. Each element in the array is -1 if incomplete or digit in [0-8], corresponding to
    digits in [1,9] 
    '''
    def __init__(self):
        with open(os.path.join(home_dir, 'data', 'easy_130k_given.p'), 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].flatten() - 1 #subtract 1 so we are zero indexed. 
        item = item.astype(int)
        return item

def get_test_puzzles(dataset_name, batch_size, device):
    '''
    Returns a tensor of starting puzzles of shape (batch_size, 81)
    '''
    if dataset_name == "larger_satnet":
        #test_dataset_sols = data.get_dataset(args.dataset, mode="test")
        full_dataset_puzzles = LargerSatNetInitial()
        test_dataset_puzzles = Subset(full_dataset_puzzles, np.arange(int(0.9*len(full_dataset_puzzles)), len(full_dataset_puzzles)))
        
        puzzles_indices = np.random.choice(len(test_dataset_puzzles), batch_size, replace=False).tolist()
        subset = Subset(test_dataset_puzzles, puzzles_indices) 
        puzzles = np.stack([subset[i] for i in range(0, len(subset))]) # (num_puzzles, 81)
        puzzles = torch.from_numpy(puzzles).to(device)
    elif dataset_name == "rrn":
        board_ds, solutions_ds = data.get_dataset(dataset_name, mode="train", with_initial_puzzles=True)
        puzzles_indices = np.random.choice(len(board_ds), batch_size, replace=False).tolist()
        subset = Subset(board_ds, puzzles_indices) 
        puzzles = torch.stack([subset[i] for i in range(0, len(subset))]).to(device) # (num_puzzles, 81)
    else:
        raise NotImplementedError()
    return puzzles 

def evaluate_samples(exp_dir, samples, epoch):
    num_valid = 0
    file_dir = os.path.join(exp_dir, 'evaluate')
    file_path = os.path.join(file_dir, 'evaluation.txt')
    if not os.path.exists(file_dir): os.makedirs(file_dir, exist_ok=True)

    with open(file_path, 'a+') as file:
        for i in range(0, len(samples)):
            if torch.is_tensor(samples):
                board = samples[i].cpu().detach().numpy().reshape((9,9)).tolist()
            elif isinstance(samples, np.ndarray):
                board = samples[i].reshape((9,9)).tolist() 
            else:
                raise ValueError()

            for row in board:
                row = [int(num) for num in row]
                row_str = ' '.join(map(str, row))
                file.write(row_str + "\n")
            
            is_valid = isValidSudoku(board)
            file.write(f'Is valid: {is_valid}\n')
            file.write('---'*50+  '\n')
            
            if is_valid: num_valid += 1
        
        print(f"Epoch: {epoch} Total boards correct: {num_valid}/{len(samples)}={num_valid/len(samples):.4f}\n")
        file.write(f"Epoch: {epoch} Total boards correct: {num_valid}/{len(samples)}={num_valid/len(samples):.4f}\n")
