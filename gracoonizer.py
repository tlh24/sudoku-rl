import math
import numpy as np
import torch as th
from torch import nn
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
		
		self.xfrmr_to_world = nn.Linear(xfrmr_dim, world_dim) 
		with th.no_grad(): 
			w = th.eye(xfrmr_dim, world_dim)
			self.xfrmr_to_world.weight.copy_( w.T )
			self.xfrmr_to_world.bias.copy_( th.zeros(world_dim) )
		
		self.softmax = nn.Softmax(dim = 2)
		self.critic_to_reward = nn.Linear(xfrmr_dim, 2) # suck in everything. 
		with th.no_grad(): 
			self.critic_to_reward.weight.copy_( th.ones(2, xfrmr_dim) / xfrmr_dim )
			self.critic_to_reward.bias.copy_( th.zeros(2) )


	def encodeBoard(self, cursPos, board, action): 
		nodes, actnodes = sudoku_to_nodes(board, cursPos, action_type)
		benc, actenc, msk = encode_nodes(nodes, nodes_act)
		return benc, actenc, msk
		
	
	def forward(self, benc, actenc, msk, n): 
		batch_size = benc.shape[0]
		board_size = benc.shape[1]
		x = th.cat((benc, actenc), axis=1)
		y,a1,a2,w1,w2 = self.xfrmr(x,msk,n)
		reward = th.ones(batch_size) * 0.05
		return y[:,:board_size,:], reward, a1, a2, w1, w2
		
	def backAction(self, benc, msk, n, newbenc, actual_action): 
		# pdb.set_trace()
		batch_size = benc.shape[0]
		actnodes = graph_encoding.sudokuActionNodes(-1) # null move.
		_,actenc,_ = graph_encoding.encode_nodes([], actnodes) 
		actenc = np.tile(actenc, [batch_size, 1, 1])
		action = th.tensor(actenc, requires_grad=True, device=benc.device)
		loss = np.zeros((1500,))
		for i in range(1500): 
			self.zero_grad()
			y,_,_,_,_,_ = self.forward(benc, action, msk, n)
			err = th.sum((y - newbenc)**2)
			err.backward()
			loss[i] = err.cpu().detach().item()
			print(loss[i])
			with th.no_grad(): 
				action -= action.grad * 0.005 # ??
				action -= action * 0.0001 # weight decay
				# action = th.clip(action, -2.5, 2.0)
			if i == 1499:
				fig, axs = plt.subplots(2, 3, figsize=(12,8))

				axs[0,2].plot(loss[:i])
				axs[0,2].set_title('loss over new board encoding')

				im = axs[0,0].imshow(newbenc[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[0,0])
				axs[0,0].set_title('target board encoding')
				
				im = axs[0,1].imshow(y[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[0,1])
				axs[0,1].set_title('predicted board encoding')
				
				im = axs[1,0].imshow(actual_action[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[1,0])
				axs[1,0].set_title('actual action')
				
				im = axs[1,1].imshow(action[0,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[1,1])
				axs[1,1].set_title('predicted action')
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
