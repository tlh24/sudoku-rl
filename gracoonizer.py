import math
import numpy as np
import torch
from torch import nn, optim
import graph_transformer_bidi as graph_transformer
import nanogpt_model
import pdb
from termcolor import colored
import matplotlib.pyplot as plt
from constants import token_cnt, g_zeroinit, g_dtype
import graph_encoding
from nanogpt_model import GPTConfig, GPT
from netdenoise import NetDenoise

USE_GRAPH_XFRMR = True
USE_NANOGPT = False

class Gracoonizer(nn.Module):
	
	def __init__(
		self,
		xfrmr_dim:int, 
		world_dim:int,
		n_heads:int,
		n_layers:int,
		repeat:int,
		mode:int
		): 
		super().__init__()
		self.xfrmr_dim = xfrmr_dim
		self.world_dim = world_dim
		self.n_head = n_heads 
		self.mode = mode
		# assert(n_layers % 4 == 0) # one layer for each of the different types.
		
		if mode == 0: # USE_GRAPH_XFRMR:
			self.xfrmr = graph_transformer.Transformer(
				d_model = xfrmr_dim,
				layers = n_layers,
				n_head = self.n_head,
				repeat = repeat,
				init_zeros = g_zeroinit
				)
		elif mode == 1: #USE_NANOGPT:
			model_args = dict(
				n_layer=n_layers,
				n_head=n_heads,
				n_embd=384,
				block_size=token_cnt,
				bias=False,
				in_size=world_dim,
				dropout=False)
			gptconf = GPTConfig(**model_args)
			self.xfrmr = nanogpt_model.GPT(gptconf)
		else: #Use NetDenoise MLP 
			# simple MLP, as a control.
			self.xfrmr = NetDenoise(token_cnt * world_dim)
	
	def forward(self, benc, hcoo):
		'''
		Given an input board, returns another board (whose shape is identical to the input board) 
		'''
		if self.mode == 0: 
			y = self.xfrmr(benc,hcoo)
		elif self.mode == 1: 
			y = self.xfrmr(benc)
		else: 
			bs = benc.shape[0]
			ntok = benc.shape[1]
			w = benc.shape[2]
			benc = torch.reshape(benc, (bs, ntok*w))
			y = self.xfrmr(benc, torch.zeros(bs, device=benc.device))
			y = torch.reshape(y, (bs, ntok, w))
		return y
		
	def backAction(self, benc, msk, n, newbenc, actual_action, lossmask, denoisenet, denoisestd):
		# record the real targets for the internal variables. 
		record_true = []
		self.forward(benc, actual_action, msk, n, record_true)
		
		batch_size = benc.shape[0]
		actnodes = graph_encoding.sudokuActionNodes(-1) # null move.
		_,actenc,_ = graph_encoding.encodeNodes([], actnodes) 
		actenc = np.tile(actenc, [batch_size, 1, 1]) # tile the null move
		action = torch.tensor(actenc, requires_grad=True, device=benc.device, dtype=g_dtype)
		opt = optim.AdamW([action], lr=1e-3, weight_decay = 5e-2)
		N = 5000
		loss = np.zeros((N,6))
		for i in range(N): 
			temp = (N - i) / (N+2.0)
			self.zero_grad()
			action.grad = None
			opt.zero_grad()
			record = []
			# pdb.set_trace()
			y,_,_,_,_,_ = self.forward(benc, action, msk, n, record)
			y = y * lossmask
			err = 10 * torch.sum((y - newbenc)**2) / np.prod(y.shape)
			err.backward(retain_graph=True)
			loss[i,0] = err.cpu().detach().item()
			x = torch.cat((benc, action), axis=1)
			losses = self.xfrmr.backAction(x, msk, newbenc, record, denoisenet, denoisestd, temp, record_true, doplot=(i==N-1))
			for j,l in enumerate(losses):
				loss[i,j+1] = l
			print(loss[i])
			opt.step()
			# with torch.no_grad(): 
				# action -= torch.nn.utils.clip_grad_norm(action, 1) * 0.05
				# action -= action.grad * 0.1 # ??
				# action -= action * 0.0001 # weight decay
				# action += torch.randn_like(action)*0.001
				# action = torch.clip(action, -2.5, 2.0) # breaks things?
			if i == N-1:
				fig, axs = plt.subplots(2, 3, figsize=(12,8))

				im = axs[0,0].imshow(newbenc[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[0,0])
				axs[0,0].set_title('target board encoding')
				
				im = axs[0,1].imshow(y[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[0,1])
				axs[0,1].set_title('predicted board encoding')
				
				yy = y - newbenc
				im = axs[0,2].imshow(yy[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[0,2])
				axs[0,2].set_title('difference: prediction - target')
				
				im = axs[1,0].imshow(actual_action[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[1,0])
				axs[1,0].set_title('actual action')
				
				im = axs[1,1].imshow(action[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[1,1])
				axs[1,1].set_title('predicted action')
				
				axs[1,2].plot(loss[:,0], "k",label="output")
				axs[1,2].plot(loss[:,1], "r",label="action")
				axs[1,2].plot(loss[:,2], "r",label="h0")
				axs[1,2].plot(loss[:,3], "g",label="h1")
				axs[1,2].plot(loss[:,4], "b",label="h2")
				axs[1,2].plot(loss[:,5], "m",label="h3")
				axs[1,2].set_title('loss over new board encoding')
				axs[1,2].legend()
				plt.show()
		return action
		
	def loadCheckpoint(self, path:str=None):
		if path is None:
			path = "checkpoints/gracoonizer.pth"
		self.load_state_dict(torch.load(path,weights_only=True))
		# if we load the state dict, then start all heads 'on'
		# self.xfrmr.allHeadsOn()
   
	def saveCheckpoint(self, path:str=None):
		if path is None:
			path = "checkpoints/gracoonizer.pth"
		torch.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")

	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Gracoonizer: number of model parameters:{trainable_params/1e6}M")
