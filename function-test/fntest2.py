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

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

'''
Goal: test if a transformer (either L1 or DP) can implement 
general functions that take an argument (or three?)
and can generalize out of training data.

as opposed to fntest.py, this one is just a pointer op.  
'''
gendata_dim = 16

def genData(bs, span): 
	# x = np.random.randn(bs, 48, gendata_dim)*1 # 32 tokens
	# # random address; amplify it a bit.
	# # makes me think of a hopfield net -- without any training!
	# # (the network is filling it in with a memory access)
	# x[:,:,-2:0] = x[:,:,-2:0] * 4 
	# x[:, :,:4] = 0 # first 4 latent dims are zero
	# x[:,-3:,:] = 0 # last 3 tokens zeroed : arg1 arg2 answer
	x = np.random.randn(bs, 48, gendata_dim)*1 # 32 tokens, 16 dims
	# add offset noise: forces the points to be in a random loc, 
	# but equidistant.
	noiz = np.random.randn(bs, 2)
	x[:,:,-1] = np.mod(np.arange(48), 7) # position encoding
	x[:,:,-1] = x[:,:,-1] + np.expand_dims(noiz[:,0], axis=1)
	x[:,:,-2] = np.arange(48) // 7 
	x[:,:,-2] = x[:,:,-2] + np.expand_dims(noiz[:,1], axis=1)
	x[:, :,:4] = 0 # first 2 latent dims are zero
	x[:,-3:,:] = 0 # last 3 tokens zeroed : arg1 arg2 answer 

	row = np.random.randint(0, span, size=bs)
	col = np.random.randint(0, span, size=bs)
	i = row * 7 + col
	y = x[np.arange(bs),i,:].copy()
	x[:,-3:,0] = 1 #  answer & arg token labels  
	x[:,-2,1] = 1 #  arg1. 
	x[:,-3,2] = 1 #  arg2.
	x[:,-2,3] = y[:,-1] # pointer address.
	x[:,-3,3] = y[:,-2] # pointer address.
	return x,y
	
	
class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
	def __init__(self, d_model: int, n_head: int):
		super().__init__()

		self.n_head = n_head
		self.d_model = d_model
		self.wk = nn.Parameter( 0.005 * torch.ones(n_head, d_model) )

		self.wqv = nn.Linear(d_model, 3*n_head*d_model)
		self.initWeights(self.wqv)
		# add in some identity
		with torch.no_grad(): 
			for i in range(3): 
				self.wqv.weight[i*d_model:(i+1)*d_model, :] += torch.eye(self.d_model, device=self.wqv.weight.device) * 0.01
			
		self.fanin = nn.Linear(d_model, d_model)

		self.l1a_f = l1attn_cuda.L1Attn()

		self.gelu = QuickGELU()
		self.rms_norm = nn.RMSNorm(d_model)

	def initWeights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.0005) # FIXME was 0.005
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)

	def attention(self, x:torch.Tensor):
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		batch_size = x.shape[0]
		ntok = x.shape[1]
		width = x.shape[2]

		v = self.wqv(x)
		v = torch.reshape(v, (batch_size, ntok, 3*self.n_head, d_head))
		q,vf,vb = torch.split(v, self.n_head, 2)

		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		wk = self.wk.unsqueeze(0).unsqueeze(0)
		k = k * wk

		# normal dense attention over all tokens
		# pad out to BLKSIZ tokens (for CUDA kernel).
		# padn = ((ntok + 15) // 16) * 16 - ntok
		# if padn == 0: 
		# 	padn = 16
		# qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		# kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		qq = q
		kk = k
		a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
		# a = a[:, :ntok+1, :ntok, :]
		# a[:, ntok, :,:] = 0.0 # slight improvement:
		# adds in e^0=1 as a 'noop' option
		# (hence max attention is 0.5, not 1)
		# a is [b,src,dst,heads]
		af = F.softmax(a, 1) # see l1attn.py -- sm over src
		ab = F.softmax(a, 2)
		# a = a[:, :ntok, :ntok, :] # remove noop
		bf = torch.einsum('bsdh, bshw -> bdhw', af, vf)
		bb = torch.einsum('bdsh, bshw -> bdhw', ab, vb) # note transpose!
		b = bf + bb
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		return b # residual sum later.
		
	def attentionDP(self, x:torch.Tensor): 
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		batch_size = x.shape[0]
		ntok = x.shape[1]

		o = self.wqv(x)
		o = torch.reshape(o, (batch_size, ntok, 3*self.n_head, d_head))
		q,k,v = torch.split(o, self.n_head, 2)
		# q,k,v are shape [batch_size, ntok, n_head, d_head]
		
		a = torch.einsum('bthw, bshw -> btsh', q, k) / math.sqrt(d_head)
		a = F.softmax(a, 1)
		b = torch.einsum('btsh, bshw -> bthw', a, v)
		b = torch.sum(b, dim=2) # sum along the heads
		return b
		

	def forward(self, x:torch.Tensor, use_dp:bool):
		if use_dp: 
			y = self.attentionDP( self.rms_norm(x) )
		else: 
			y = self.attention(x)
		y = self.gelu(y)
		y = self.fanin(y) # allow sign inversions & mixing; no dim change
		return x + y

class Transformer(nn.Module):
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.resblocks = nn.ModuleList(\
			[ResidualAttentionBlock(d_model, n_head) \
				for _ in range(layers)])
		self.in_proj = nn.Linear(gendata_dim, d_model, bias=True)
		self.out_proj = nn.Linear(d_model, gendata_dim, bias=True)

	def forward(self, x:torch.Tensor, use_dp:bool):
		# x is dtype int to interface with the embedding layer
		bs,n_tok,inw = x.shape
		x = self.in_proj(x)
		# x = torch.cat((x, torch.zeros(bs, n_tok, self.d_model - inw, device=x.device)), axis=-1)
		for i in range(self.repeat):
			for j, layer in enumerate(self.resblocks):
				x = layer(x, use_dp)
		return self.out_proj(x)

	def fixedInit(self):
		for layer in self.resblocks:
			layer.fixedInit()

	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params}")
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', action='store_true', help='make test data and plot it')
	parser.add_argument('-b', type=int, default=128, help='batch size')
	parser.add_argument('-c', action='store_true', help='start fresh, dont load a model')
	parser.add_argument('-a', action='store_true', help='use AdamW')
	parser.add_argument('-v', action='store_true', help='validate only')
	parser.add_argument('-d', action='store_true', help='dot product attention')
	cmd_args = parser.parse_args()
	
	if cmd_args.t: 
		batch_size = 1
		x, y = genData(batch_size, 6)
		fig,axs = plt.subplots(1,2)
		axs[0].imshow(np.squeeze(x))
		axs[1].imshow(y)
		plt.show()
		exit()

	batch_size = cmd_args.b
	
	model = Transformer(d_model=64, layers=2, repeat=1, n_head=2)
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
		optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, amsgrad=True)
	else: 
		optimizer = psgd.LRA(model.parameters(),\
			lr_params=0.01,lr_preconditioner= 0.01, momentum=0.9,\
			preconditioner_update_probability=0.25, \
			exact_hessian_vector_product=False, \
			rank_of_approximation=20, grad_clip_max_norm=5.0)
	
	fd_losslog = open('losslog.txt', 'w')
	
	input_thread = threading.Thread(target=utils.monitorInput, daemon=True)
	input_thread.start()
	
	def train(uu):
		x,y = genData(16*2048, 5)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()

		for i in range(24*2000): # num iters
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
		x,y = genData(4*2048, 6)
		x = torch.tensor(x).float()
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()
		
		for i in range(4*2048 // batch_size):
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
