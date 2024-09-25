import random
import numpy as np
import torch
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PuzzleSolutionDataset(Dataset):
    def __init__(self, boards, solutions):
        '''
        boards: numpy.NdArray (num_samples, 81)
        solutions: numpy.NdArray (num_samples, 81)

        get_item returns board, solution which has shape ((81,) , (81,) )
        '''
        self.boards = boards 
        self.solutions = solutions 
        assert self.boards.shape == self.solutions.shape 
            
    def __len__(self):
        return len(self.boards)
            
    def __getitem__(self, idx):
        return torch.from_numpy(self.boards[idx]), torch.from_numpy(self.solutions[idx])