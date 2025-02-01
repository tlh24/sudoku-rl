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
general functions that take an argument (or two?) 
and can generalize out of the training data.

Input data is maximally-tokenized 4x4 mini-sudoku. 
This is designed to look like a tagged datastructure, ala JSON. 
	(a C datastructure would have bare values and pointers)
c:1,axis:1:1,axis:2:1,axis:3:1; c:0,axis:1:1,axis:2:2,axis:3:1
c:2,axis:1:1,axis:2:2,axis:3:2; c:3,axis:1:1,axis:2:4,axis:3:2
axis 1 is row
axis 2 is column
axis 3 is block
Hence these are addresses

Integers are encoded directly.  
Vocabulary is encoded with learnable vectors.  

Goal is implement hasA(axis,index,digit) 
'''

def genData(bs, puzzl): 
	indx_offset = 1
	x = np.zeros((bs, 16 * 12, 4), dtype=int)
	for row in range(4): 
		for col in range(4): 
			i = (row*4 + col) * 11
			x[:, i, 1] = 5+1 # 'cell'
			x[:, i+1 , 1] = puzzl[:,row,col]+1 # *word* encoding
			x[:, i+2 , 1] = 5+2 # 'axis'
			x[:, i+3 , 0] = 1 # axis 1
			x[:, i+4 , 0] = row + indx_offset # position
			x[:, i+5 , 1] = 5+2 # 'axis'
			x[:, i+6 , 0] = 2 # axis 2
			x[:, i+7 , 0] = col + indx_offset # position
			x[:, i+8 , 1] = 5+2 # 'axis'
			x[:, i+9 , 0] = 3  # axis 3
			x[:, i+10, 0] = (row // 2)*2 + (col // 2) + indx_offset # block
	
	y = np.zeros((bs,))
	for b in range(bs): 
		# provide the arguments to the function. 
		axis = np.random.randint(3) + 1
		# axis = 1 # FIXME
		index = np.random.randint(4) 
		# index = 1 # FIXME
		digit = np.random.randint(4) + 1
		# digit = 1 # FIXME
		x[b,-7,1] = 5+3 # 'request'
		x[b,-6,1] = 5+2 # 'axis'
		x[b,-5,0] = axis
		x[b,-4,0] = index + indx_offset # was + 0
		x[b,-3,1] = 5+1 # 'cell'
		x[b,-2,1] = digit # word encoding
		x[b,-1,1] = 5+4 # 'answer'
		# now, calculate it.  
		if axis == 1: 
			y[b] = np.sum(puzzl[b, index, :] == digit)
		if axis == 2: 
			y[b] = np.sum(puzzl[b, :, index] == digit)
		if axis == 3: 
			r = index // 2
			c = index % 2
			m = puzzl[b, r:r+2, c:c+2]
			y[b] = np.sum(m == digit)
		# make it simple as a control - count the number of 'digits' in the first row. 
		# y[b] = np.sum(puzzl[b, :, :] == digit) # FIXME
			
	# add in position encoding -- without any structural hints
	x[:,:,-1] = np.arange(16*12)
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
		self.fanin = nn.Linear(d_model, d_model)

		self.l1a_f = l1attn_cuda.L1Attn()

		self.gelu = QuickGELU()
		self.rms_norm = nn.RMSNorm(d_model)

	def initWeights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.005) # FIXME
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
		padn = ((ntok + 15) // 16) * 16 - ntok
		if padn == 0: 
			padn = 16
		qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
		a = a[:, :ntok+1, :ntok, :]
		a[:, ntok, :,:] = 0.0 # slight improvement:
		# adds in e^0=1 as a 'noop' option
		# (hence max attention is 0.5, not 1)
		# a is [b,src,dst,heads]
		a = F.softmax(a, 1) # see l1attn.py -- sm over src
		a = a[:, :ntok, :ntok, :] # remove noop
		bf = torch.einsum('bsdh, bshw -> bdhw', a, vf)
		bb = torch.einsum('bdsh, bshw -> bdhw', a, vb) # note transpose!
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
		self.in_proj = nn.Linear(d_model, d_model, bias=True)
		self.out_proj = nn.Linear(d_model, 1, bias=True)
		self.embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=d_model-4)

	def forward(self, x:torch.Tensor, use_dp:bool):
		# x is dtype int to interface with the embedding layer
		bs,n_tok,_ = x.shape
		x_flat = x[:,:,1].view(-1)
		embed = self.embedding_layer(x_flat)
		x = torch.cat((x.float(), embed.view(bs,n_tok,self.d_model-4)), dim=-1)
		x = self.in_proj(x)
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
	# batch_size = 1
	# puzzl = np.random.randint(5, size=(batch_size,4,4))
	# x, y = genData(batch_size, puzzl)
	# print(puzzl)
	# print(x)
	# print(y)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', type=int, default=128, help='batch size')
	parser.add_argument('-c', action='store_true', help='start fresh, dont load a model')
	parser.add_argument('-a', action='store_true', help='use AdamW')
	parser.add_argument('-v', action='store_true', help='validate only')
	parser.add_argument('-d', action='store_true', help='dot product attention')
	cmd_args = parser.parse_args()

	batch_size = cmd_args.b
	
	model = Transformer(d_model=64, layers=3, repeat=3, n_head=4)
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
		
	model = nn.DataParallel(model)
	model = model.cuda()

	if cmd_args.a: 
		optimizer = optim.AdamW(model.module.parameters(), lr=2.5e-4, amsgrad=True)
	else: 
		optimizer = psgd.LRA(model.module.parameters(),\
			lr_params=0.01,lr_preconditioner= 0.01, momentum=0.9,\
			preconditioner_update_probability=0.25, \
			exact_hessian_vector_product=False, \
			rank_of_approximation=20, grad_clip_max_norm=5.0)
	
	fd_losslog = open('losslog.txt', 'w')
	
	input_thread = threading.Thread(target=utils.monitorInput, daemon=True)
	input_thread.start()
	
	def train(uu):
		puzzl = np.random.randint(5, size=(16*2048,4,4))
		x,y = genData(16*2048, puzzl)
		x = torch.tensor(x) # leave as int
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()

		for i in range(24*2000):
			indx = torch.randperm(x.shape[0])
			indx = indx[:batch_size]
			xx = x[indx,:,:]
			target = y[indx]

			if cmd_args.a: 
				optimizer.zero_grad()
				pred = model(xx, cmd_args.d)
				loss = torch.sum( (pred[:,-1,0] - target)**2 )
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				loss.backward()
				optimizer.step()
			else: 
				def closure():
					pred = model(xx, cmd_args.d)
					# only look at the last token
					loss = torch.sum( (pred[:,-1,0] - target)**2 ) + \
						sum( \
							[torch.sum(5e-4 * torch.rand_like(param) * torch.abs(param) ) \
						for param in model.module.parameters()])
					return loss
				loss = optimizer.step(closure)
			lloss = loss.detach().cpu().item()
			if i % 10 == 0:
				print(lloss)
				fd_losslog.write(f'{uu}\t{lloss}\n')
				fd_losslog.flush()
			uu += 1
			if uu % 1000 == 0: 
				torch.save(model.module.state_dict(), 'fntest.pt')
				print(colored('saved model', 'blue'))
			if utils.switch_to_validation:
				break
			
		return uu
		
	def test(uu): 
		puzzl = np.random.randint(5, size=(2048,4,4))
		x,y = genData(2048, puzzl)
		x = torch.tensor(x) # leave as int
		y = torch.tensor(y).float()
		x = x.cuda()
		y = y.cuda()
		
		for i in range(2048 // batch_size):
			indx = torch.arange(i*batch_size, (i+1)*batch_size)
			xx = x[indx,:,:]
			target = y[indx]
			pred = model(xx, cmd_args.d)
			loss = torch.sum( (pred[:,-1,0] - target)**2 )
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
