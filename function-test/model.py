import math
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

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
	def __init__(self, d_model: int, n_head: int):
		super().__init__()

		self.n_head = n_head
		self.d_model = d_model
		self.wk = nn.Parameter( torch.ones(n_head, d_model) )

		self.wqv = nn.Linear(d_model, 3*n_head*d_model)
		self.wqkv = nn.Linear(d_model, 4*n_head*d_model) 
		# self.initWeights(self.wqv)
		# # add in some identity
		# with torch.no_grad():
		# 	for i in range(3):
		# 		self.wqv.weight[i*d_model:(i+1)*d_model, :] += torch.eye(self.d_model, device=self.wqv.weight.device) * 0.01

		self.fanin = nn.Linear(d_model, d_model)

		self.l1a_f = l1attn_cuda.L1Attn()

		self.gelu = QuickGELU()
		self.rms_norm = nn.RMSNorm(d_model)
		self.layer_norm = nn.LayerNorm(d_model)

	def initWeights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)

	def attention(self, x:torch.Tensor):
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		bs = x.shape[0]
		ntok = x.shape[1]
		width = x.shape[2]

		# zscore data along *all* dimensions first
		x = (x - torch.mean(x)) / (1*torch.std(x))
		v = self.wqv(x)
		v = torch.reshape(v, (bs, ntok, 3*self.n_head, d_head))
		q,vf,vb = torch.split(v, self.n_head, 2)

		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		wk = self.wk.unsqueeze(0).unsqueeze(0)
		k = k * wk
		q = q * wk # TEST test! definite improvement!   
		# k = self.layer_norm(k) # only norm the k - allow the Q's to float.
		# layerNorm does not seem to help.  N=4

		# normal dense attention over all tokens
		# pad out to BLKSIZ tokens (for CUDA kernel).
		# padn = ((ntok + 15) // 16) * 16 - ntok
		# if padn == 0:
		# 	padn = 16
		# qq = torch.cat((q, torch.zeros(bs, padn, n_head, width, device=v.device)), axis=1)
		# kk = torch.cat((k, torch.zeros(bs, padn, n_head, width, device=v.device)), axis=1)
		qq = q # shape bs, ntok, nhead, width
		kk = k
		a = self.l1a_f(qq, kk) # includes -1 / sqrt(head)
			# a <= 0 by construction, so that softmax works.
		a = a * -1 * math.sqrt(width) # reverse the scaling!
		ac = 0.7071 * ( \
			torch.sum(torch.abs(qq), axis=3).unsqueeze(1).expand(bs,ntok,ntok,n_head) + \
			torch.sum(torch.abs(kk), axis=3).unsqueeze(2).expand(bs,ntok,ntok,n_head) )
			# i think this is correct..
		# figs,axs = plt.subplots(2,3)
		# im = axs[0,0].imshow(a[0,:,:,0].cpu().detach().numpy())
		# plt.colorbar(im,ax=axs[0,0])
		# axs[0,0].set_title('a')
		# im = axs[0,1].imshow(ac[0,:,:,0].cpu().detach().numpy())
		# plt.colorbar(im,ax=axs[0,1])
		# axs[0,1].set_title('ac')

		a = (a - ac)/10 # idk...
		pdb.set_trace()

		# im = axs[0,2].imshow(a[0,:,:,0].cpu().detach().numpy())
		# plt.colorbar(im,ax=axs[0,2])
		# axs[0,2].set_title('a - ac')
  #
		# im = axs[1,0].imshow(qq[0,:,0,:].cpu().detach().numpy())
		# plt.colorbar(im,ax=axs[1,0])
		# axs[1,0].set_title('qq')
		# im = axs[1,1].imshow(kk[0,:,0,:].cpu().detach().numpy())
		# plt.colorbar(im,ax=axs[1,1])
		# axs[1,1].set_title('kk')
		# plt.show()

		if False:
			figs,axs = plt.subplots(2,2)
			wq,wvf,wvb = torch.split(self.wqv.weight, d_head*n_head, 0)
			im = axs[0,0].imshow(wq.cpu().detach().numpy())
			plt.colorbar(im,ax=axs[0,0])
			axs[0,0].set_title('WQ')
			im = axs[0,1].imshow(wvf.cpu().detach().numpy())
			plt.colorbar(im,ax=axs[0,1])
			axs[0,1].set_title('WVF')
			im = axs[1,0].imshow(wvb.cpu().detach().numpy())
			plt.colorbar(im,ax=axs[1,0])
			axs[1,0].set_title('WVB')
			axs[1,1].plot(self.wk.squeeze().cpu().detach().numpy())
			axs[1,1].set_title('WK')
			plt.show()

			figs,axs = plt.subplots(2,2)
			axs[0,0].plot(q[0,0,0,:].detach().cpu().numpy(), label='Q tok 0')
			axs[0,0].plot(k[0,0,0,:].detach().cpu().numpy(), label='K tok 0')
			axs[0,0].legend()
			axs[0,1].plot(q[0,-1,0,:].detach().cpu().numpy(), label='Q tok -1')
			axs[0,1].plot(k[0,-1,0,:].detach().cpu().numpy(), label='K tok -1')
			axs[0,1].legend()
			axs[1,0].plot(q[0,-2,0,:].detach().cpu().numpy(), label='Q tok -2')
			axs[1,0].plot(k[0,-2,0,:].detach().cpu().numpy(), label='K tok -2')
			axs[1,0].legend()
			axs[1,1].plot(q[0,-3,0,:].detach().cpu().numpy(), label='Q tok -3')
			axs[1,1].plot(k[0,-3,0,:].detach().cpu().numpy(), label='K tok -3')
			axs[1,1].legend()
			plt.show()
			# attention in the heads.
			fig, axs = plt.subplots(1,n_head)
			if n_head > 1:
				for h in range(n_head):
					a0 = a[20, :, :, h].squeeze().cpu().detach().numpy()
					im = axs[h].imshow(a0)
					axs[h].set_title(f'attention head {h}')
					plt.colorbar(im,ax=axs[h])
			else:
				a0 = a[20, :, :, 0].squeeze().cpu().detach().numpy()
				im = axs.imshow(a0)
				axs.set_title(f'attention head 0')
				plt.colorbar(im,ax=axs)
			plt.show()
		# a = a - 0.5*torch.mean(a, dim=(1,2)).unsqueeze(1).unsqueeze(1) # makes very little difference, surprisingly: might be doing it wrong?

		# try making the nonlinearity smoother - zero derivative at origin.
		# a = -1*torch.exp(-1*a*a) * a*a + (1-torch.exp(-1*a*a))*a
		# also seems to make very little difference.

		# a = a[:, :ntok+1, :ntok, :]
		# a[:, ntok, :,:] = 0.0 # slight improvement:
		# adds in e^0=1 as a 'noop' option
		# (hence max attention is 0.5, not 1)
		# a is [b,src,dst,heads]
		af = F.softmax(a, 1) # see l1attn.py -- softmax over src
		ab = F.softmax(a, 2) # fix feb 2025: softmax over dst
		# a = a[:, :ntok, :ntok, :] # remove noop
		bf = torch.einsum('bsdh, bshw -> bdhw', af, vf)
		bb = torch.einsum('bsdh, bdhw -> bshw', ab, vb)
				# eliminate over dest (the softmax dim)
		b = bf + bb
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (bs, ntok, self.d_model))
		return b # residual sum later.

	def attentionDP(self, x:torch.Tensor):
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		batch_size = x.shape[0]
		ntok = x.shape[1]

		o = self.wqkv(x)
		o = torch.reshape(o, (batch_size, ntok, 4*self.n_head, d_head))
		q,k,vf,vb = torch.split(o, self.n_head, 2)
		# q,k,v are shape [batch_size, ntok, n_head, d_head]

		a = torch.einsum('bthw, bshw -> btsh', q, k) / math.sqrt(d_head)
		af = F.softmax(a, 2) # sm over eliminated s dim
		bf = torch.einsum('btsh, bshw -> bthw', af, vf)
		ab = F.softmax(a, 1) # sm over eliminated t dim
		bb = torch.einsum('btsh, bthw -> bshw', af, vb)
		b = bf + bb
		b = torch.sum(b, dim=2) # sum along the heads
		return b


	def forward(self, x:torch.Tensor, use_dp:bool):
		if use_dp:
			y = self.attentionDP( self.rms_norm(x) )
		else:
			y = self.attention( x )
		y = self.gelu(y)
		y = self.fanin(y) # allow sign inversions & mixing; no dim change
		return x + y

class Transformer(nn.Module):
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int, gendata_dim:int):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.resblocks = nn.ModuleList(\
			[ResidualAttentionBlock(d_model, n_head) \
				for _ in range(layers)])
		win = torch.cat( (torch.eye(gendata_dim), torch.zeros(d_model - gendata_dim, gendata_dim) ), dim=0)
		# self.in_proj = nn.Parameter( win )
		self.in_proj = nn.Linear(gendata_dim, d_model, bias=False)
		self.out_proj = nn.Linear(d_model, gendata_dim, bias=True)

	def forward(self, x:torch.Tensor, use_dp:bool):
		# x is dtype int to interface with the embedding layer
		bs,n_tok,inw = x.shape
		x = self.in_proj(x)
		# x = torch.einsum("btg,dg->btd", x, self.in_proj)
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
