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
        Eps is uniformly sampled in range [eps_low, eps_high] and applied to all tokens in board sequence

    Returns: (one hot encoded board, value score)
    '''
    def __init__(self, value_dataset, is_noisy, eps_low, eps_high):
        self.value_dataset = value_dataset
        self.is_noisy = is_noisy
        self.eps_low = eps_low 
        self.eps_high = eps_high
         
    def __len__(self):
        return len(self.value_dataset)
    
    def __getitem__(self, idx):
        board_tens, value = self.value_dataset[idx] #board_tens is sequence of ints 0-8, value is float in [0,1]
        one_hot_tens = F.one_hot(board_tens, num_classes=9)
        if self.is_noisy:
            eps = np.random.uniform(self.eps_low, self.eps_high)
            smoothed_tens = noise_one_hot(one_hot_tens, eps)
            return (smoothed_tens, value)
        return (one_hot_tens, value)
    

def num_constraints_violated(board_tensor):
    '''
    Returns the number of constraints violated, where iterate over all rows, cols, and blocks and count number of pairs 

    board_tensor:  has shape (81,) or (1,81)
    '''
    if torch.is_tensor(board_tensor):
        board_seq = board_tensor.cpu().detach().numpy()
    elif isinstance(board_tensor, np.ndarray):
        board_seq = board_tensor
    else:
        raise ValueError()

    board_matrix = board_seq.reshape((9,9))

    def count_pairs(group):
        pairs = 0
        for i in range(9):
            for j in range(i + 1, 9):
                if group[i] == group[j]:
                    pairs += 1
        return pairs

    total_pairs = 0

    # Count pairs in rows
    for i in range(0, len(board_matrix)):
        row = board_matrix[i].tolist()
        total_pairs += count_pairs(row)

    # Count pairs in columns
    for j in range(0, board_matrix.shape[1]):
        col = board_matrix[:, j].tolist()
        total_pairs += count_pairs(col)

    # Count pairs in 3x3 blocks
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = [board_matrix[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            total_pairs += count_pairs(block)

    return total_pairs

    
class ValueDataset(Dataset):
    '''
    Given some dataset containing num_samples many tensors of completed sudoku puzzles,
    
    returns tuples (board_tensor, value_score) where the value_score is defined by num_constraints violated 

        value_score = 1 - num_constraints_violated/ max_num_constraints_violated
    '''
    def __init__(self, tens_dataset, num_samples=100000):
        self.tens_dataset = tens_dataset
        self.num_samples = min(num_samples, len(self.tens_dataset))
        self.boards = torch.zeros(self.num_samples, 81, dtype=torch.long)
        self.constraints_violated = np.zeros(self.num_samples)
     
        for i in range(0, self.num_samples):
            board_solution = self.tens_dataset[i]
            corrupted_board_solution = board_solution.detach().clone()
            # randomly corrupt this board by replacing digits with random number 
            num_cells_to_corrupt = np.random.choice(60)
            for idx in np.random.choice(81, size=num_cells_to_corrupt, replace=False):
                corrupted_board_solution[idx] = np.random.choice(9)
            self.boards[i] = corrupted_board_solution

            self.constraints_violated[i] = num_constraints_violated(corrupted_board_solution)

        max_constraints_violated = np.max(self.constraints_violated)
        normalized_values = 1 - self.constraints_violated/max_constraints_violated #value scores in [0,1]
        self.values = normalized_values

    def __len__(self):
        return len(self.tens_dataset)

    def __getitem__(self, idx):
        # returns (board_tensor, value_float)
        return (self.boards[idx], self.values[idx])

def get_one_hot_dataset(dataset_name, is_noisy, eps_low, eps_high):
    '''
    Returns a (noisy) one_hot encoding of the board digits
        
    eps_low: (float) The lowest eps to be used in label smoothing 
    eps_high: (float) Highest eps to be used in label smoothing 
    '''
    tens_dataset = get_dataset(dataset_name, "train")
    value_dataset = ValueDataset(tens_dataset, 1000)
    breakpoint()
    return OneHotNoisy(value_dataset, is_noisy, eps_low, eps_high)  


if __name__ == "__main__":
    ds = get_one_hot_dataset("rrn", True, 0.01, 0.1)
    first = ds[0]
    breakpoint()
    print("Completed") 