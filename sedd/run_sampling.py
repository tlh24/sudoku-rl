import torch
from load_model import load_model_local
import argparse 
import sampling 
import data_utils as data 
import numpy as np 
from torch.utils.data import Subset
import os 
home_dir = os.path.dirname(os.path.dirname(__file__))
import pickle 
from torch.utils.data import Subset
from utils import isValidSudoku


class LargerSatNetInitial:
    '''
    Returns initial puzzles as a 1d array. Each element in the array is digit in [0-8], corresponding to
    digits in [1,9], or -1 if incomplete
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


def main(args):
    device = torch.device('cuda')
    model, graph, noise = load_model_local('./',device, args.model_path, args.checkpoint_num)
    if args.evaluate:
        #test_dataset_sols = data.get_dataset(args.dataset, mode="test")
        full_dataset_puzzles = LargerSatNetInitial()
        test_dataset_puzzles = Subset(full_dataset_puzzles, np.arange( int(0.9*len(full_dataset_puzzles)), len(full_dataset_puzzles)))

        puzzles_indices = np.random.choice(len(test_dataset_puzzles), args.batch_size, replace=False).tolist()
        subset = Subset(test_dataset_puzzles, puzzles_indices) #(num_puzzles, 81)
        puzzles = np.stack([subset[i] for i in range(0, len(subset))])
        puzzles = torch.from_numpy(puzzles).to(device)
        
        def proj_fun(x: torch.Tensor):
            '''
            Replaces each tensor and infills the puzzle with initial hints
            x: tensor of shape (num_puzzles, 81). 
            '''
            infilled_x = torch.where(puzzles > -1, puzzles, x)
            return infilled_x
    

        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (args.batch_size, args.seq_len), 'analytic', args.steps, device=device, proj_fun=proj_fun)

        samples = proj_fun(sampling_fn(model))
        num_valid = 0
        file_dir = os.path.join(args.model_path, 'evaluate')
        file_path = os.path.join(file_dir, 'evaluation.txt')
        if not os.path.exists(file_dir): os.makedirs(file_dir, exist_ok=True)

        with open(file_path, 'w+') as file:
            for i in range(0, len(samples)):
                board = samples[i].cpu().detach().numpy().reshape((9,9))
                for row in board:
                    row = [int(num) for num in row]
                    row_str = ' '.join(map(str, row))
                    file.write(row_str + "\n")
                
                is_valid = isValidSudoku(board)
                file.write(f'Is valid: {is_valid}\n')
                file.write('---'*50+  '\n')
                
                if is_valid: num_valid += 1
            
            print(f"Total boards correct: {num_valid}/{len(samples)}={num_valid/len(samples):.2f}\n")
            file.write(f"Total boards correct: {num_valid}/{len(samples)}\n")


            



    else:
        raise NotImplementedError() 
 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='experiments/09-05-2024-13:15')
    parser.add_argument("--checkpoint_num", type=int, required=True)
    parser.add_argument("--dataset", type=str, default='larger_satnet')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument('--seq_len', type=int, default=81)
    args = parser.parse_args()
    main(args)


