'''Inference driver code'''
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
import pdb


def main(args):
	device = torch.device('cuda')
	model, graph, noise = load_model_local('./',device, args.model_path, args.checkpoint_num)

	if args.dataset == "larger_satnet":
		test_dataset_puzzles, test_dataset_sols = data.get_dataset(args.dataset, mode="test", with_initial_puzzles=True)
		
		puzzles_indices = np.random.choice(len(test_dataset_puzzles), args.num_to_eval, replace=False).tolist()
		subset = Subset(test_dataset_puzzles, puzzles_indices) # (num_puzzles, 81)
		if isinstance(subset[i], np.ndarray):
			puzzles = np.stack([subset[i] for i in range(0, len(subset))])
			puzzles = torch.from_numpy(puzzles).to(device)
		elif torch.is_tensor(subset[i]):
			puzzles = torch.stack([subset[i] for i in range(0, len(subset))]).to(device)
		else:
			raise ValueError("Dataset must be np array or tensors")
		
	elif args.dataset == "rrn":
		board_ds, solutions_ds = data.get_dataset(args.dataset, mode="train", with_initial_puzzles=True)
		puzzles_indices = np.random.choice(len(board_ds), args.num_to_eval, replace=False).tolist()
		subset = Subset(board_ds, puzzles_indices) # (num_puzzles, 81)
		puzzles = torch.stack([subset[i] for i in range(0, len(subset))]).to(device)
	elif args.dataset == "satnet":
		board_ds, solutions_ds = data.get_dataset(args.dataset, mode="test", with_initial_puzzles=True)
		puzzles_indices = np.random.choice(len(board_ds), args.num_to_eval, replace=False).tolist()
		subset = Subset(board_ds, puzzles_indices) # (num_puzzles, 81)
		puzzles = torch.stack([subset[i] for i in range(0, len(subset))]).to(device) 
	else:
		raise NotImplementedError()
	
	def proj_fun(x: torch.Tensor):
		'''
		Replaces each tensor and infills the puzzle with initial hints that have values [0,8]
		x: tensor of shape (num_puzzles, 81). 
		Note: digits should be values [0,8] corresponding to values [1,9]
		
		'''
		infilled_x = torch.where(puzzles > -1, puzzles, x)
		return infilled_x


	sampling_fn = sampling.get_pc_sampler(
		graph, noise, (args.num_to_eval, args.seq_len), 'analytic', args.steps, device=device, proj_fun=proj_fun)

	samples = proj_fun(sampling_fn(model))
	num_valid = 0
	file_dir = os.path.join(args.model_path, 'evaluate')
	file_path = os.path.join(file_dir, 'evaluation.txt')
	if not os.path.exists(file_dir): os.makedirs(file_dir, exist_ok=True)

	with open(file_path, 'w+') as file:
		for i in range(0, len(samples)):
			def printBoard(brd, check=False): 
				brd = np.clip(brd+1, 0, 9)
				nonlocal num_valid
				full_str = ""
				for row in brd:
					row = [int(num) for num in row]
					row_str = ' '.join(map(str, row))
					full_str = full_str + "".join(map(str, row))
					file.write(row_str + "\n")
				file.write(full_str + "\n") # for checking w online tools
				if check: 
					is_valid = isValidSudoku(brd)
					file.write(f'Is valid: {is_valid}\n')
					if is_valid: 
						num_valid += 1
			board = samples[i].cpu().detach().numpy().reshape((9,9))
			puzz = puzzles[i].cpu().detach().numpy().reshape((9,9))
			file.write('-----   puzzle: -----\n')
			printBoard(puzz)
			file.write('----- solution: -----\n')
			printBoard(board, check=True)
			file.write('.....................\n')
		
		print(f"Total boards correct: {num_valid}/{len(samples)}={num_valid/len(samples):.4f}\n")
		file.write(f"Total boards correct: {num_valid}/{len(samples)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='experiments/10-22-2024-15:05')
    parser.add_argument("--checkpoint_num", type=int, required=True)
    parser.add_argument("--dataset", type=str, default='rrn')
    parser.add_argument("--num_to_eval", type=int, default=512) #number of puzzles to evaluate
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument('--seq_len', type=int, default=81)
    args = parser.parse_args()
    main(args)


