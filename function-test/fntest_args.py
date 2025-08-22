import sys
import os
import math
import argparse
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored
from torch import nn, optim
import l1attn_cuda
import matplotlib.pyplot as plt
import psgd
import pdb
import threading
import multiprocessing
from model import Transformer

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

'''
Goal: test if a transformer (either L1 or DP) can implement 
general functions that take an argument (or three?)
and can generalize out of training data.

as opposed to fntest.py, this one is just a pointer op:
	given two x & y locs (as address tokens),
	retrieve the (random) data at that position.
	Data is uniformly spaced, with a random offset per datapoint.
	Position is linearly encoded.
'''
gendata_dim = 8
indicator = 3

def genData(bs):
	x = np.zeros((bs, 4, 8))
	y = np.zeros((bs, 8))
	# diagonal for the indicators
	x[:,np.arange(4),np.arange(4)] = indicator
	# make the args
	x[:,:-1,-1] = np.random.randn(bs,3) * 2
	bsar = np.arange(bs)
	y = x[bsar,-1,:]
	# transpose to 'grab' the args
	y[bsar,4:7] = x[bsar,:3,-1]
	return x,y


def positiveControl(x): 
	# make sure the task can be done 'manually'. 
	bs = x.shape[0]
	y = x[:,-1,:].squeeze()
	bsar = np.arange(bs)
	y[bsar,4:7] = x[bsar,:3,-1]
	return y
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', action='store_true', help='make test data and plot it')
	parser.add_argument('-b', type=int, default=128, help='batch size')
	parser.add_argument('-c', action='store_true', help='start fresh, dont load a model')
	parser.add_argument('-a', action='store_true', help='use AdamW')
	parser.add_argument('-v', action='store_true', help='validate only')
	parser.add_argument('-d', action='store_true', help='dot product attention')
	parser.add_argument('-l', type=str, default='', help='losslog label')
	parser.add_argument('-u', type=int, default=0, help='CUDA device')
	parser.add_argument('-n', type=int, default=25000, help='number of gradient steps')
	parser.add_argument('--doplot', action='store_true', help="plot internal model activations")
	cmd_args = parser.parse_args()
	
	if cmd_args.t: 
		x, y = genData(256)
		pred = positiveControl(x)
		print("positive control error:", np.sum( (y - pred)**2 ) )
		batch_size = 1
		x, y = genData(batch_size)
		fig,axs = plt.subplots(1,2)
		im = axs[0].imshow(np.squeeze(x))
		plt.colorbar(im, ax=axs[0])
		im = axs[1].imshow(y)
		plt.colorbar(im, ax=axs[1])
		plt.show()
		exit()

	batch_size = cmd_args.b
	
	model = Transformer(d_model=16, layers=1, repeat=1, n_head=3, gendata_dim=gendata_dim)
	model.printParamCount()
	if cmd_args.c: 
		print(colored("not loading any model weights.", "blue"))
	else: 
		try: 
			model.load_state_dict(\
				torch.load('fntest.pt',weights_only=True,map_location='cpu'))
			print(colored("loaded model.", "green"))
		except Exception as error:
			print(error)
		
	# model = nn.DataParallel(model)
	model = model.to(f"cuda:{cmd_args.u}")

	if cmd_args.a: 
		optimizer = optim.AdamW(model.parameters(), lr=5e-4, amsgrad=False, weight_decay=0.01, betas=(0.9,0.99) )
	else: 
		optimizer = psgd.LRA(model.parameters(),\
			lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,\
			preconditioner_update_probability=0.25, \
			exact_hessian_vector_product=False, \
			rank_of_approximation=20, grad_clip_max_norm=5.0)
	
	fd_losslog = open(f'losslog_{cmd_args.l}.txt', 'w')
	
	input_thread = threading.Thread(target=utils.monitorInput, daemon=True)
	input_thread.start()
	
	def train(uu):
		x,y = genData(2048)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.to(f"cuda:{cmd_args.u}")
		y = y.to(f"cuda:{cmd_args.u}")

		for i in range(cmd_args.n): # num iters
			indx = torch.randperm(x.shape[0])
			indx = indx[:batch_size]
			xx = x[indx,:,:]
			target = y[indx]

			if cmd_args.a: 
				optimizer.zero_grad()
				pred = model(xx, cmd_args.d, cmd_args.doplot)
				loss = torch.sum( (pred[:,-1,:] - target)**2 )
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				loss.backward()
				optimizer.step()
			else: 
				def closure():
					pred = model(xx, cmd_args.d, cmd_args.doplot)
					# only look at the last token
					loss = torch.sum( (pred[:,-1,:] - target)**2 ) + \
						sum( \
							[torch.sum(1e-2 * torch.rand_like(param) * torch.abs(param) ) \
						for param in model.parameters()])
					return loss
				loss = optimizer.step(closure)
			lloss = loss.detach().cpu().item()
			if i % 10 == 0:
				print(lloss)
				fd_losslog.write(f'{uu}\t{lloss}\n')
				fd_losslog.flush()
			uu += 1
			if uu % 1000 == 0: 
				torch.save(model.state_dict(), 'fntest.pt')
				print(colored('saved model', 'blue'))
			if utils.switch_to_validation:
				break
			
		return uu
		
	def test(uu): 
		test_size = 32*2048
		x,y = genData(test_size)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.to(f"cuda:{cmd_args.u}")
		y = y.to(f"cuda:{cmd_args.u}")
		
		for i in range(test_size // batch_size):
			indx = torch.arange(i*batch_size, (i+1)*batch_size)
			xx = x[indx,:,:]
			target = y[indx]
			pred = model(xx, cmd_args.d, cmd_args.doplot)
			loss = torch.sum( (pred[:,-1,:] - target)**2 )
			lloss = loss.detach().cpu().item()
			print('v',lloss)
			fd_losslog.write(f'{uu}\t{lloss}\n')
			fd_losslog.flush()
			uu += 1

	uu = 0
	if not cmd_args.v: 
		uu = train(uu)
	test(uu)
	
	fd_losslog.close()
