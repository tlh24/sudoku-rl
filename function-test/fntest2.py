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
gendata_dim = 24

def genData(bs, span): 
	# create random data vectors:
	indicator = 10
	x = np.random.randn(bs, 48, gendata_dim)*1 # 32 tokens, 16 dims
	# add offset noise: forces the points to be in a random loc, 
	# but equidistant.
	noiz = np.random.randn(bs, 2)*1.5
	x[:,:,-1] = np.mod(np.arange(48), 7) # position encoding
	x[:,:,-1] = x[:,:,-1] + np.expand_dims(noiz[:,0], axis=1)
	x[:,:,-2] = np.arange(48) // 7 
	x[:,:,-2] = x[:,:,-2] + np.expand_dims(noiz[:,1], axis=1)
	x[:, :,:5] = 0 # first 6 latent dims are zero
	x[:,-3:,:] = 0 # last 3 tokens zeroed : arg1 arg2 answer 

	row = np.random.randint(0, span, size=bs)
	col = np.random.randint(0, span, size=bs)
	i = row * 7 + col
	y = x[np.arange(bs),i,:].copy()
	x[:,-3,0] = indicator # arg1
	# x[:,-1,0] = indicator*2 # output
	x[:,-2,1] = indicator # arg2.
	# x[:,-1,1] = indicator*2 # output
	x[:,-1,2] = indicator # output only
	x[:,-2,3] = y[:,-1] # pointer address.
	x[:,-3,3] = y[:,-2] # pointer address.
	x[:,-2,4] = y[:,-1] # pointer address.
	x[:,-3,4] = y[:,-2] # pointer address.
	y[:,:-2] = x[:,-1,:-2] # copy everything but the pointer loc
	# print(y[0,:])
	# plt.imshow(x[0,:,:])
	# plt.show()
	return x,y
	
def positiveControl(x): 
	# make sure the task can be done 'manually'. 
	bs = x.shape[0]
	ntok = x.shape[1]
	y = np.zeros((bs,gendata_dim))
	for b in range(bs): 
		p1 = x[b,-2,4]
		p2 = x[b,-3,4]
		targ = np.zeros((ntok,2))
		targ[:,0] = p2
		targ[:,1] = p1
		dist = np.sum(np.abs(x[b,:,-2:] - targ), axis=1)
		indx = np.argmin(dist)
		y[b,:] = x[b,indx,:]
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
	cmd_args = parser.parse_args()
	
	if cmd_args.t: 
		x, y = genData(256, 6)
		pred = positiveControl(x)
		print("positive control error:", np.sum( (y - pred)**2 ) )
		batch_size = 1
		x, y = genData(batch_size, 6)
		fig,axs = plt.subplots(1,2)
		axs[0].imshow(np.squeeze(x))
		axs[1].imshow(y)
		plt.show()
		exit()

	batch_size = cmd_args.b
	
	model = Transformer(d_model=64, layers=1, repeat=1, n_head=2, gendata_dim=gendata_dim)
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
	model = model.cuda()

	if cmd_args.a: 
		optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, amsgrad=True, weight_decay=0.01)
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
		x,y = genData(24*2048, 5)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()

		for i in range(25000): # num iters
			indx = torch.randperm(x.shape[0])
			indx = indx[:batch_size]
			xx = x[indx,:,:]
			target = y[indx]

			if cmd_args.a: 
				optimizer.zero_grad()
				pred = model(xx, cmd_args.d)
				loss = torch.sum( (pred[:,-1,:] - target)**2 )
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				loss.backward()
				optimizer.step()
			else: 
				def closure():
					pred = model(xx, cmd_args.d)
					# only look at the last token
					loss = torch.sum( (pred[:,-1,:] - target)**2 ) + \
						sum( \
							[torch.sum(5e-4 * torch.rand_like(param) * torch.abs(param) ) \
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
		x,y = genData(test_size, 6)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()
		
		for i in range(test_size // batch_size):
			indx = torch.arange(i*batch_size, (i+1)*batch_size)
			xx = x[indx,:,:]
			target = y[indx]
			pred = model(xx, cmd_args.d)
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
