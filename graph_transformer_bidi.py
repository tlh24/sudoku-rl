import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchlayers as tl
import l1attn_sparse_bidi_cuda
import l1attn_cuda
import pdb
import matplotlib.pyplot as plt
from constants import g_l1atten, SuN, g_dtype, token_cnt

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class StraightThroughQuantize(torch.autograd.Function):
	# see https://www.kaggle.com/code/peggy1502/learning-pytorch-2-new-autograd-functions/notebook
	# and https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
	@staticmethod
	def forward(ctx, input):
		y = torch.round(input*10.0) / 10.0
		return y

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output
	  
# class LinearM(nn.Module):
# 	# with the bias merged -- used for PSGD optimizer.
# 	def __init__(self, indim:int, outdim:int, initzeros:bool):
# 		super(LinearM, self).__init__()
# 		scl = 0.005
# 		if initzeros:
# 			self.w = torch.nn.Parameter( scl * torch.ones(outdim,indim+1,dtype=g_dtype))
# 		else:
# 			self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim+1,dtype=g_dtype))
# 		with torch.no_grad():
# 			self.w[:,-1] = 0.0 # bias starts at 0
#
# 	def forward(self, x):
# 		return torch.einsum('oi,bhi -> bho', self.w[:,:-1], x) + self.w[:,-1]
#
# class LinearNobias(nn.Module):
# 	def __init__(self, indim:int, outdim:int, initzeros:bool):
# 		super(LinearNobias, self).__init__()
# 		scl = 0.02 / math.sqrt(2 * 9)
# 		if initzeros:
# 			scl = 0.0
# 		self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim,dtype=g_dtype) )
#
# 	def forward(self, x):
# 		return torch.einsum('oi,bhi -> bho', self.w[:, :-1], x)
# 		# einsum is slower than mm (!)

class ResidualAttentionBlock(nn.Module): 
	def __init__(self, d_model: int, n_head: int):
		super().__init__()
		
		init_zeros = False
		self.n_head = n_head
		self.d_model = d_model
		self.wk = nn.Parameter( 0.005 * torch.ones(n_head, d_model) )

		self.wqv = nn.Linear(d_model, 3*n_head*d_model)
		self.initWeights(self.wqv)
		self.fanin = nn.Linear(d_model, d_model)
		self.initWeights(self.wqv)
		
		self.l1a_s = l1attn_sparse_bidi_cuda.L1AttnSparseBidi()
		self.l1a_f = l1attn_cuda.L1Attn()

		self.gelu = QuickGELU()
		# self.gelu = nn.ReLU() # slightly worse, unlike the l1-attn example.
		
	def initWeights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.005) # FIXME
			if module.bias is not None:
					torch.nn.init.zeros_(module.bias)


	def attention(self, x:torch.Tensor, hcoo:list, layer:int, pas:int):
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		batch_size = x.shape[0]
		ntok = x.shape[1]
		width = x.shape[2]
		
		# q = self.wq(x)
		# q = torch.reshape(q, (batch_size, ntok, self.n_head, d_head))
		# v = self.wv(x)
		# v = torch.reshape(v, (batch_size, ntok, 2*self.n_head, d_head))
		# vf,vb = torch.split(v, self.n_head, 2)

		v = self.wqv(x)
		v = torch.reshape(v, (batch_size, ntok, 3*self.n_head, d_head))
		q,vf,vb = torch.split(v, self.n_head, 2)
		
		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		
		wk = self.wk.unsqueeze(0).unsqueeze(0)
		k = k * wk
		
		# cycle through the coo vectors.  
		if hcoo[layer % len(hcoo)] == 'dense':
			# normal dense attention over all tokens
			# pad out to BLKSIZ tokens (for CUDA kernel).
			padn = ((ntok + 15) // 16) * 16 - ntok
			assert(padn > 0) # for noop
			qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
			kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
			a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
			a = a[:, :ntok+1, :ntok, :]
			a[:, ntok, :,:] = 0.0 # slight improvement..
			# add in e^0=1 as a 'noop' option
			# (hence max attention is 0.5, not 1)
			# output is b,src,dst,heads
			a = F.softmax(a, 1) # see l1attn.py -- sm over src
			a = a[:, :ntok, :ntok, :] # remove noop
			bf = torch.einsum('bsdh, bshw -> bdhw', a, vf)
			bb = torch.einsum('bdsh, bshw -> bdhw', a, vb)
			b = bf + bb
		elif torch.is_tensor(hcoo[layer % len(hcoo)]):
			# must be a global / top-level token coordinate vec
			# extract these tokens and pass to dense l1 attn
			a2a = hcoo[layer % len(hcoo)]
			a2len = a2a.shape[0]
			q = q[:,a2a,:,:]
			k = k[:,a2a,:,:]
			vf = vf[:,a2a,:,:]
			vb = vb[:,a2a,:,:]
			# pad out to BLKSIZ tokens (for CUDA kernel).
			padn = ((a2len + 15) // 16) * 16 - a2len
			assert(padn > 0) # for noop
			qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
			kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
			a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
			a = a[:, :a2len+1, :a2len, :]
			a[:, a2len, :,:] = 0.0
			# add in e^0=1 as a 'noop' option
			# (hence max attention is 0.5, not 1)
			# output is b,src,dst,heads
			a = F.softmax(a, 1) # see l1attn.py -- sm over src
			a = a[:, :a2len, :a2len, :] # remove noop
			bf = torch.einsum('bsdh, bshw -> bdhw', a, vf)
			bb = torch.einsum('bdsh, bshw -> bdhw', a, vb)
			# scatter to original sites
			b = torch.zeros(batch_size, ntok, n_head, width, device=v.device)
			indx = torch.arange(0, a2len, device=v.device)
			b[:,a2a,:,:] = bf[:,indx,:,:] + bb[:,indx,:,:]
		elif hcoo[layer % len(hcoo)] == 'self':
			# token attends to itself.
			# do it manually, no need for slow indexing in the sparse lib.
			a = torch.abs(q - k).sum(dim=-1) / (-1*math.sqrt(self.d_model))
			a = torch.exp(a)
			a = a / (a+1) # softmax with a zero option
			b = vf * a[..., None]
		else:
			# sparse attention.
			coo,dst_mxlen = hcoo[layer % len(hcoo)] 
			use_softmax = True 
			b = self.l1a_s(vf,vb,q,k,coo,dst_mxlen,use_softmax)
		
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		 
		return b # residual sum later.

	# @torch.compile
	def forward(self, x:torch.Tensor, hcoo:list, layer:int, pas:int):
		y = self.attention(x,hcoo,layer,pas)
		# y = self.fanout(y)
		y = self.gelu(y+SuN/2.0)-(SuN/2.0) # this nonlinearity is essential
		# y = self.gelu(y)
		y = self.fanin(y) # allow sign inversions & mixing; no dim change
		# y = self.gelu(y) # this destroys performance! 
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
		self.stq = StraightThroughQuantize()
		self.in_proj = nn.Linear(d_model, d_model, bias=True)
		self.out_proj = nn.Linear(d_model, d_model, bias=True)

	# @torch.compile
	def forward(self, x:torch.Tensor, hcoo:list):
		# x = self.in_proj(x)
		for i in range(self.repeat): 
			# x = self.stq.apply(x) # quantize FIXME
			for j, layer in enumerate(self.resblocks):
				# linearly encode the repeat position on all tokens. 
				# x[:,:,0] = i*2 FIXME
				x = layer(x,hcoo,j,i)
			x[:,:,32:] = 0.0 # clear internal encoding FIXME
		return x # self.out_proj(x)
