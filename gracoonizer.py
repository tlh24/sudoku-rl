import math
import numpy as np
import torch as th
from torch import nn, optim
import graph_transformer
import pdb
from termcolor import colored
import matplotlib.pyplot as plt
from constants import n_heads, g_zeroinit
import graph_encoding


class Gracoonizer(nn.Module):
	
	def __init__(
		self,
		xfrmr_dim:int, 
		world_dim:int,
		reward_dim:int
		): 
		super().__init__()
		self.xfrmr_dim = xfrmr_dim
		self.world_dim = world_dim
		self.reward_dim = reward_dim
		self.n_head = n_heads # need one head for each of the 4 connection types.
		
		# self.world_to_xfrmr = nn.Linear(world_dim, xfrmr_dim)
		# with th.no_grad(): 
		# 	w = th.cat([th.eye(world_dim, world_dim) for _ in range(self.n_head)], 0)
		# 	self.world_to_xfrmr.weight.copy_( w )
		# 	self.world_to_xfrmr.bias.copy_( th.zeros(xfrmr_dim) )
		
		self.gelu = graph_transformer.QuickGELU()
		
		self.xfrmr = graph_transformer.Transformer(
			d_model = xfrmr_dim, 
			layers = 2, # was 2
			n_head = self.n_head, 
			repeat = 1, # was 3
			init_zeros = g_zeroinit
			)
		
		# self.xfrmr_to_world = graph_transformer.LinearM(xfrmr_dim, world_dim, True) 
		# with th.no_grad(): 
		# 	w = th.zeros(world_dim, xfrmr_dim+1)
		# 	for i in range(min(xfrmr_dim, world_dim)): 
		# 		w[i,i] = 1.0
		# 	self.xfrmr_to_world.w.copy_( w )
		
		# self.softmax = nn.Softmax(dim = 2)
		# self.critic_to_reward = graph_transformer.LinearM(xfrmr_dim, 2, True) # suck in everything. 
		# with th.no_grad(): 
		# 	self.critic_to_reward.w.copy_( th.ones(2, xfrmr_dim+1) / xfrmr_dim )


	def encodeBoard(self, cursPos, board, action): 
		nodes, actnodes = sudoku_to_nodes(board, cursPos, action_type)
		benc, actenc, msk = encodeNodes(nodes, nodes_act)
		return benc, actenc, msk
		
	
	def forward(self, benc, actenc, msk, n, record): 
		batch_size = benc.shape[0]
		board_size = benc.shape[1]
		if record is not None: 
			record.append(actenc)
		x = th.cat((benc, actenc), axis=1)
		y,a1,a2,w1,w2 = self.xfrmr(x,msk,n,record)
		reward = th.ones(batch_size) * 0.05
		return y[:,:board_size,:], reward, a1, a2, w1, w2
		
	def backAction(self, benc, msk, n, newbenc, actual_action, lossmask, denoisenet, denoisestd):
		# record the real targets for the internal variables. 
		record_true = []
		self.forward(benc, actual_action, msk, n, record_true)
		
		batch_size = benc.shape[0]
		actnodes = graph_encoding.sudokuActionNodes(-1) # null move.
		_,actenc,_ = graph_encoding.encodeNodes([], actnodes) 
		actenc = np.tile(actenc, [batch_size, 1, 1]) # tile the null move
		action = th.tensor(actenc, requires_grad=True, device=benc.device)
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
			err = 10 * th.sum((y - newbenc)**2) / np.prod(y.shape)
			err.backward(retain_graph=True)
			loss[i,0] = err.cpu().detach().item()
			x = th.cat((benc, action), axis=1)
			losses = self.xfrmr.backAction(x, msk, newbenc, record, denoisenet, denoisestd, temp, record_true, doplot=(i==N-1))
			for j,l in enumerate(losses):
				loss[i,j+1] = l
			print(loss[i])
			opt.step()
			# with th.no_grad(): 
				# action -= th.nn.utils.clip_grad_norm(action, 1) * 0.05
				# action -= action.grad * 0.1 # ??
				# action -= action * 0.0001 # weight decay
				# action += th.randn_like(action)*0.001
				# action = th.clip(action, -2.5, 2.0) # breaks things?
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
		
	def load_checkpoint(self, path:str=None):
		if path is None:
			path = "checkpoints/gracoonizer.pth"
		self.load_state_dict(th.load(path))
		# if we load the state dict, then start all heads 'on'
		self.xfrmr.allHeadsOn()
   
	def save_checkpoint(self, path:str=None):
		if path is None:
			path = "checkpoints/gracoonizer.pth"
		th.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")

	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params/1e6}M")
