import math
import numpy as np
import torch as th
from torch import nn
import glip_model
import pdb
from termcolor import colored
import matplotlib.pyplot as plt
import graph_model

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
		self.n_head = 3*4 # need one head for each of the 4 connection types.
		
		# self.world_to_xfrmr = nn.Linear(world_dim, xfrmr_dim)
		# with th.no_grad(): 
		# 	w = th.cat([th.eye(world_dim, world_dim) for _ in range(self.n_head)], 0)
		# 	self.world_to_xfrmr.weight.copy_( w )
		# 	self.world_to_xfrmr.bias.copy_( th.zeros(xfrmr_dim) )
		
		self.gelu = glip_model.QuickGELU()
		
		self.xfrmr = glip_model.Transformer(
			d_model = xfrmr_dim, 
			layers = 2, # was 2
			n_head = self.n_head, 
			repeat = 1, # was 3
			init_zeros = False
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
		nodes = sudoku_to_nodes(board, cursPos, action_type)
		enc, msk = encode_nodes(nodes)
		return enc, msk
		
	
	def forward(self, enc, msk, n): 
		# note: spatially, the number of latents = number of actions
		batch_size = enc.shape[0]
		# x = self.world_to_xfrmr(x) # optional gelu here.
		y,a1,a2,w1,w2 = self.xfrmr(enc,msk,n)
		# for softmax, have to allow for "no action"=[0] and "no number"=[18]
		# action = th.cat( 
		# 	(self.softmax(action[:,:,0:10]), self.softmax(action[:,:,10:])), 2)
		
		# given the board and suggested action, predict the reward. 
		# aslow = self.action_slow.unsqueeze(0).expand(batch_size, -1, -1)
		# action_l = th.cat((aslow, action), 2)
		# w = self.gelu(self.world_to_xfrmr(board_enc))
		# x = th.cat((w, y[:,nt:,:]), 1) # token dim
		# z = self.critic(x)
		# reward = self.critic_to_reward(y[:,nt:,:]) # one reward for each action
		reward = th.ones(batch_size) * 0.05
		return y, reward, a1, a2, w1, w2
		
	def load_checkpoint(self, path:str=None):
		if path is None:
			path = "checkpoints/gracoonizer.pth"
		self.load_state_dict(th.load(path))
   
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
