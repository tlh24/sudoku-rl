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

In particular: test the pointer op: access an addressed
(possibly computed in a different layer) token.
'''

def graycodePosEnc(ntok, nbits, rand_phase=False):
	'''
	Generate a graycode position encoding
	'''
	pos_enc = np.zeros((ntok,nbits*2), dtype=np.float32)
	indx = np.linspace(0, (ntok-1)*2*math.pi, ntok)
	if rand_phase:
		phase_offset = np.random.uniform() * 2 * math.pi
	else:
		phase_offset = 0
	for i in range(nbits):
		# gray code: [0][1] has a period of 4
		# [2][3] has a period of sqrt(4*8) = sqrt(32) = 4 sqrt(2)
		# [4][5] period of 8..
		# period = 4 * (math.sqrt(2.0))**i
		#   above is slower frequency progression - does not help?
		period = 4 * 2.0**i
		if True:
			# sinusoidal, seems to work better?
			pos_enc[:, 2*i  ] = -np.cos(indx / period + phase_offset)
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset)
		else:
			# graycode! (thresholded)
			pos_enc[:, 2*i  ] = np.cos(indx / period + phase_offset) < 0
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset) < 0
	return pos_enc

def randomPosEnc(ntok, nbits, rand_phase=False):
	return np.random.randn(ntok, nbits*2)

def genData(bs, span): 
	assert(span < 31)
	indicator = 10
	x = np.random.randn(bs, 32, 16)*1 # 32 tokens, 16 dims
	x[:, :,:2] = 0 # first 2 latent dims are zero
	x[:,-1,:] = 0 # last token zeroed / answer
	x[:,-1,0] = indicator #  answer token.  model is very sensitive to this: larger works better. (why?)
	x[:,:,-1] = np.arange(32) # position encoding

	i = np.random.randint(0, span, size=bs) + (32 - span)//2
	y = x[np.arange(bs),i,:].copy()
	x[:,-1,1] = i # pointer address.
	x[:,-1,-1] = 0 # position encoding.
	return x,y
	
def genDataGraycode(bs, span):
	assert(span < 31)
	indicator = 10
	nbits = 6
	x = np.random.randn(bs, 32, 16+nbits*2)*1 # 32 tokens, 16 dims
	x[:, :,:2] = 0 # first 2 latent dims are zero
	x[:,-1,:] = 0 # last token zeroed / answer
	x[:,-1,0] = indicator #  answer token.  model is very sensitive to this: larger works better. (why?)
	for b in range(bs):
		posenc = graycodePosEnc(32, nbits, rand_phase=True)
		x[b,:,-12:] = posenc # position encoding

	i = np.random.randint(0, span, size=bs) + (32 - span)//2
	y = x[np.arange(bs),i,:].copy()
	x[:,-1,-12:] = x[np.arange(bs),i,-12:] # pointer address.
	return x,y
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', type=int, default=128, help='batch size')
	parser.add_argument('-c', action='store_true', help='start fresh, dont load a model')
	parser.add_argument('-a', action='store_true', help='use AdamW')
	parser.add_argument('-v', action='store_true', help='validate only')
	parser.add_argument('-d', action='store_true', help='dot product attention')
	parser.add_argument('-l', type=str, default='', help='losslog label')
	parser.add_argument('-t', action='store_true', help='test genData')
	parser.add_argument('--graycode', action='store_true', help='use graycode pos enc')
	parser.add_argument('--headslayers', type=int, default=1, help='number of heads and layers')
	cmd_args = parser.parse_args()

	batch_size = cmd_args.b
	
	if cmd_args.t:
		batch_size = 1
		if cmd_args.graycode:
			x, y = genDataGraycode(batch_size, 16)
		else:
			x, y = genData(batch_size, 16)
		fig,axs = plt.subplots(1,2)
		axs[0].imshow(np.squeeze(x))
		axs[0].set_title('X')
		axs[1].imshow(y)
		axs[1].set_title('Y')
		plt.show()
		exit()

	gendata_dim = 16
	if cmd_args.graycode:
		gendata_dim += 12
	model = Transformer(d_model=64, layers=cmd_args.headslayers, repeat=1, n_head=cmd_args.headslayers, gendata_dim=gendata_dim)
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
			lr_params=0.01,lr_preconditioner= 0.01, momentum=0.9,\
			preconditioner_update_probability=0.25, \
			exact_hessian_vector_product=False, \
			rank_of_approximation=20, grad_clip_max_norm=5.0)
	
	prefix = "l1"
	if cmd_args.d:
		prefix = "dp"
	if cmd_args.graycode:
		prefix += "_gcPE"
	prefix += f"_h{cmd_args.headslayers}l{cmd_args.headslayers}"
	fd_losslog = open(f'losslog_{prefix}_{cmd_args.l}.txt', 'w')
	
	input_thread = threading.Thread(target=utils.monitorInput, daemon=True)
	input_thread.start()
	
	def train(uu):
		if cmd_args.graycode:
			x,y = genDataGraycode(16*2048, 16)
		else:
			x,y = genData(16*2048, 16)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()

		for i in range(15*2000): # num iters
			indx = torch.randperm(x.shape[0])
			indx = indx[:batch_size]
			xx = x[indx,:,:]
			target = y[indx]

			if cmd_args.a: 
				optimizer.zero_grad()
				pred = model(xx, cmd_args.d, doplot=False)
				loss = torch.sum( (pred[:,-1,:] - target)**2 )
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				loss.backward()
				optimizer.step()
			else: 
				def closure():
					pred = model(xx, cmd_args.d, doplot=False)
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
				if not cmd_args.c:
					torch.save(model.state_dict(), 'fntest.pt')
					print(colored('saved model', 'blue'))
			if utils.switch_to_validation:
				break
			
		return uu
		
	def test(uu): 
		if cmd_args.graycode:
			x,y = genDataGraycode(16*2048, 30)
		else:
			x,y = genData(16*2048, 30)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()
		
		for i in range(16*2048 // batch_size):
			indx = torch.arange(i*batch_size, (i+1)*batch_size)
			xx = x[indx,:,:]
			target = y[indx]
			pred = model(xx, cmd_args.d, doplot=False)
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
