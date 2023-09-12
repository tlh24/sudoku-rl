import math
import torch as th
from torch import nn
import torch.cuda.amp
from ctypes import *
import clip_model
from pathlib import Path
from typing import Union


class Racoonizer(nn.Module):
	
	def __init__(
		self,
		xfrmr_width:int, 
		world_dim:int,
		latent_cnt:int,
		action_dim:int,
		reward_dim:int
		): 
		super().__init__()
		self.xfrmr_width = xfrmr_width
		self.world_dim = world_dim
		self.latent_cnt = latent_cnt
		self.action_dim = action_dim
		self.reward_dim = reward_dim
		
		self.world_to_xfrmr = nn.Linear(world_dim, xfrmr_width)
		self.gelu = clip_model.QuickGELU()
		
		self.model = clip_model.Transformer(
			width = xfrmr_width, 
			layers = 4, 
			heads = 4, 
			attn_mask = None)
		
		self.model_to_action = nn.Linear(xfrmr_width, action_dim)
		self.softmax = nn.Softmax(dim = 0)
		self.model_to_reward = nn.Linear(xfrmr_width, 2) 
			# reward: immediate and infinite-horizon
		self.latent_slow = nn.Parameter(
			th.randn(latent_cnt, world_dim // 2) / (world_dim ** 0.5))
		
		
	def encodePos(self, i, j): 
		p = th.zeros(6)
		scl = 2 * math.pi / 9.0
		p[0] = math.sin(i*scl)
		p[1] = math.cos(i*scl)
		p[2] = math.sin(j*scl)
		p[3] = math.cos(j*scl)
		block = i // 3 + (j // 3) * 3
		p[4] = math.sin(block*scl) # slightly cheating here
		p[5] = math.cos(block*scl)
		return p


	def encodeBoard(self, cursPos, board, guess, notes): 
		x = th.zeros(1 + 81, self.world_dim)
		
		# encode the cursor token
		x[0, 0] = 1
		x[0, 1+9*3:] = self.encodePos(cursPos[0], cursPos[1])

		#encode the board state
		for i in range(9): 
			for j in range(9): 
				k = 1 + i*9 + j # token number
				m = math.floor(board[i][j]) # works b/c 1 indexed.
				if m > 0: 
					x[k, m] = 1.0
				m = math.floor(guess[i][j]) # also ok: 1-indexed.
				if m > 0: 
					x[k, m+9] = 1.0
				x[k, 1+9*2:1+9*3] = notes[i,j,:]
				x[k,1+9*3:] = self.encodePos(i, j)
		return x
	
	def forward(self, board_enc, latents): 
		latents = th.cat((self.latent_slow, latents), 1)
		x = th.cat((board_enc, latents), 0)
		x = self.gelu(self.world_to_xfrmr(x))
		y = self.model(x)
		action = self.model_to_action(y[0,:])
		# softmax over numbers and actions
		action = th.cat( 
			(self.softmax(action[0:10]), self.softmax(action[10:])), 0)
		reward = self.model_to_reward(y[0,:])
		return action, reward
		
	def backLatent(self, board_enc, action, reward): 
		# in supervised learning need to derive latent based on 
		# action, reward, and board state. 
		latents = torch.zeros(latent_cnt, world_dim // 2, requires_grad = True)
		ap, rp = self.forward(board_enc, latents)
		err = th.sum((action - ap)**2, (reward - rp)**2)
		err.backward()
		print(latents.grad())
		pdb.set_trace()
		latents += latents.grad() * 0.1 # ??
		return latents

	def print_n_params(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params/1e6}M")
