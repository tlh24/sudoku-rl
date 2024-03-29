import math
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pdb
import matplotlib.pyplot as plt
import graph_encoding
from gracoonizer import Gracoonizer
from sudoku_gen import Sudoku
from plot_mmap import make_mmf, write_mmap
from netdenoise import NetDenoise
from constants import *
from type_file import Action
from tqdm import tqdm
import psgd 


class BinaryDataset(Dataset):
	'''
	Dataset where the ith element is a sample containing random vector,
		zeros/ones vector, zero/one int, graph_mask, board_reward
      If int is zero, needs to predict zeros vector; else ones vector
	'''
	def __init__(self, num_samples, enc_dim=20):
		self.orig_boards_enc = torch.randint(low=0,high=10, size=(num_samples,enc_dim))    
      actions = torch.randint(low=0, high=2, size=(num_samples, 1))
      new_state = actions.repeat(1,20)
      # zero pad action enc to be length enc_dim
      actions = F.pad(actions, (0, enc_dim-1), "constant", 0)
		self.actions_enc = actions 
      self.new_boards_enc = new_state
		self.graph_masks = graph_masks
		self.rewards = torch.zeros(num_samples) 

	def __len__(self):
		return self.orig_boards_enc.size(0)

	def __getitem__(self, idx):
		sample = {
			'orig_board' : self.orig_boards_enc[idx],
			'new_board' : self.new_boards_enc[idx], 
			'action_enc' : self.actions_enc[idx],
			'graph_mask' : self.graph_masks[idx],
			'reward' : self.rewards[idx].item(),
		}
		return sample

def getTestDataLoaders(num_samples, num_eval=2000):
	'''
	Returns a train and test dataloader for a mock dataset what has 
	an 2d vector, a binary number (0 or 1), and corresponding vector of 
   zeros or ones
	'''
	
	



if __name__ == '__main__':
	NUM_SAMPLES = 12000
	NUM_EVAL = 2000
	NUM_EPOCHS = 500
	device = torch.device('cuda:0')
	fd_losslog = open('testlosslog.txt', 'w')
	args = {"NUM_SAMPLES": NUM_SAMPLES, "NUM_EPOCHS": NUM_EPOCHS, "NUM_EVAL": NUM_EVAL,\
			 "device": device, "fd_losslog": fd_losslog}
	optimizer_name = "adam"
	
	torch.set_float32_matmul_precision('high')
	torch.manual_seed(42)
	
	# get our train and test dataloaders
	train_dataloader, test_dataloader = getDataLoaders(puzzles, args["NUM_SAMPLES"], args["NUM_EVAL"])
	
	# allocate memory
	memory_dict = getMemoryDict()
	
	# define model 
	model = Gracoonizer(xfrmr_dim = 20, world_dim = 20, reward_dim = 1).to(device)
	model.printParamCount()
	try: 
		model.load_checkpoint()
		print("loaded model checkpoint")
	except : 
		print("could not load model checkpoint")
	
	optimizer = getOptimizer(optimizer_name, model)
	criterion = nn.MSELoss()

	epoch_num = 0
	for _ in tqdm(range(0, args["NUM_EPOCHS"])):
		train(args, memory_dict, model, train_dataloader, optimizer, criterion, epoch_num)
		epoch_num += 1
	
	# save after training
	model.save_checkpoint()

	print("validation")
	validate(args, model, test_dataloader, criterion, epoch_num)