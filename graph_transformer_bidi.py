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
	  
class LinearM(nn.Module): 
	# with the bias merged -- used for PSGD optimizer.
	def __init__(self, indim:int, outdim:int, initzeros:bool): 
		super(LinearM, self).__init__()
		scl = 0.005
		if initzeros: 
			self.w = torch.nn.Parameter( scl * torch.ones(outdim,indim+1,dtype=g_dtype))
		else:
			self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim+1,dtype=g_dtype))
		with torch.no_grad(): 
			self.w[:,-1] = 0.0 # bias starts at 0
		
	def forward(self, x):
		return torch.einsum('oi,bhi -> bho', self.w[:,:-1], x) + self.w[:,-1]
		
class LinearNobias(nn.Module): 
	def __init__(self, indim:int, outdim:int, initzeros:bool): 
		super(LinearNobias, self).__init__()
		scl = 0.02 / math.sqrt(2 * 9)
		if initzeros: 
			scl = 0.0
		self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim,dtype=g_dtype) )
		
	def forward(self, x):
		return torch.einsum('oi,bhi -> bho', self.w[:, :-1], x)
		# einsum might be slower than mm

class ResidualAttentionBlock(nn.Module): 
	def __init__(self, d_model: int, n_head: int):
		super().__init__()
		
		init_zeros = False
		self.n_head = n_head
		self.d_model = d_model
		self.wq = LinearM(d_model, n_head*d_model, init_zeros) # constant init works fine, just a bit slower. 
		self.wv = LinearM(d_model, 2*n_head*d_model, init_zeros)
		#self.wqv = LinearM(d_model, n_head*2*d_model, init_zeros) 
		self.wk = torch.nn.Parameter( 0.005 * torch.ones(n_head, d_model) )
		
		self.l1a_s = l1attn_sparse_bidi_cuda.L1AttnSparseBidi()
		self.l1a_f = l1attn_cuda.L1Attn()
		self.soft = torch.nn.Softmax(dim=2) # unused with L1 attn
		self.fanout = LinearM(d_model, d_model * 1, False) # non-zero init
		self.gelu = QuickGELU()
		# self.fanin = nn.Linear(d_model * 3, d_model)
		
	def attention(self, x:torch.Tensor, hcoo:list, n:int, layer:int, pas:int, record=list):
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		width = x.shape[2]
		
		q = self.wq(x)
		q = torch.reshape(q, (batch_size, ntok, self.n_head, d_head))
		v = self.wv(x)
		v = torch.reshape(v, (batch_size, ntok, 2*self.n_head, d_head))
		vf,vb = torch.split(v, self.n_head, 2)
		
		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		
		wk = self.wk.unsqueeze(0).unsqueeze(0)
		k = k * wk # + bk # with bias to allow for centering.
		
		if g_l1atten: 
			# cycle through the coo vectors.  
			if hcoo is None:
				# !only all-to-all layers.
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
				a = F.softmax(a, 1) # see l1attn.py -- sm over src FIXME 1
				a = a[:, :ntok, :ntok, :] # remove noop
				bf = torch.einsum('bsdh, bshw -> bdhw', a, vf)
				bb = torch.einsum('bdsh, bshw -> bdhw', a, vb)
				b = bf + bb
			elif layer % 4 == 3:
				# extract all global / all-to-all tokens
				# really could do this with pure sparse attn.. will have to compare.
				a2a = hcoo[3]
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
			else: 
				# sparse attention.
				coo,dst_mxlen = hcoo[layer%4] 
				use_softmax = True 
				b = self.l1a_s(vf,vb,q,k,coo,dst_mxlen,use_softmax)
			ap = torch.zeros(ntok, ntok, n_head) # dummy.
		else: 
			# DP attention
			a = torch.einsum('bthd,bshd -> btsh', q, k) / math.sqrt(d_head)
			a = self.soft(a)
			# a = a * msk 
			b = torch.einsum('btsh,bshd -> bthd', a, v) # regular attention
			# ap = (a[0,:,:,:] - 1.0 + msk[0,:,:,:]).squeeze().detach().cpu()
			ap = (a[0,:,:,:]).squeeze().detach().cpu()
		
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		 
		return b # residual sum later.

	def forward(self, x:torch.Tensor, hcoo:list, n:int, layer:int, pas:int, record=list):
		y = self.attention(x,hcoo,n,layer,pas,record)
		if record is not None: 
			record.append( y )
		# y = self.fanout(y)
		y = self.gelu(y+SuN/2.0)-(SuN/2.0) # this nonlinearity is essential
		# y = self.gelu(y)
		y = self.fanout(y) # allow sign inversions & mixing; no dim change
		# y = self.gelu(y) # this destroys performance! 
		return x + y
		
		
class Transformer(nn.Module): 
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int, init_zeros:bool):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.resblocks = nn.ModuleList([ResidualAttentionBlock(d_model, n_head) for _ in range(layers)])

	def forward(self, x:torch.Tensor, hcoo:list, n:int, record:list):
		for i in range(self.repeat): 
			for j, layer in enumerate(self.resblocks):
				# linearly encode the repeat position on all tokens. 
				x[:,:,0] = i*2
				x = layer(x,hcoo,n,j,i,record)
		return x
