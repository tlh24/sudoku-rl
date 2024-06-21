'''
Train our nano-GPT model.
Boiler-plate code adapted from https://github.com/azreasoners/recurrent_transformer/blob/main/sudoku/main.py#L11
'''
from utils import set_seed, Sudoku_Dataset_SATNet
import torch


def main(args=None):
    set_seed(42)

    ###Load Data###
    dataset = Sudoku_Dataset_SATNet()
    indices = list(range(len(dataset)))

    test_dataset = torch.utils.data.Subset(dataset, indices[-1000:])
    train_dataset = torch.utils.data.Subset(dataset, indices[:min(9000, args.n_train)])

    ###Build GPT model and trainer###
    model_conf = {}
    

main()

