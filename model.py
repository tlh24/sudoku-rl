import math
import numpy as np
import torch as th
from torch import nn
import blip_model
import pdb
from termcolor import colored
import matplotlib.pyplot as plt

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
		self.num_tokens = 81+1
		
		self.world_to_xfrmr = nn.Linear(world_dim, xfrmr_width)
		self.gelu = blip_model.QuickGELU()
		
		self.xfrmr = blip_model.Transformer(
			d_model = xfrmr_width, 
			layers = 2, # was 2
			n_head = 4, 
			repeat = 3, # was 3
			init_zeros = True
			)
		
		self.xfrmr_to_world = nn.Linear(xfrmr_width, world_dim)
		self.xfrmr_to_action = nn.Linear(xfrmr_width, action_dim)
		
		# self.world_to_critic = nn.Linear(world_dim, xfrmr_width)
		
		# self.critic = clip_model.Transformer(
		# 	width = xfrmr_width, 
		# 	layers = 2, 
		# 	heads = 4, 
		# 	repeat = 1, 
		# 	attn_mask = None)
		
		self.softmax = nn.Softmax(dim = 2)
		self.critic_to_reward = nn.Linear(xfrmr_width, 2) 
			
		self.latent_slow = nn.Parameter(
			th.randn(latent_cnt, world_dim // 2) / (world_dim ** 0.5))
		action_slow_dim = world_dim - action_dim
		self.action_slow = nn.Parameter(
			th.randn(latent_cnt, action_slow_dim) / (action_slow_dim ** 0.5))
		
		
	def encodePos(self, i, j): # row, column
		p = np.zeros(8)
		scl = 2 * math.pi / 9.0
		p[0] = math.sin(i*scl)
		p[1] = math.cos(i*scl)
		p[2] = math.sin(j*scl)
		p[3] = math.cos(j*scl)
		block = (i // 3)*3 + (j // 3)
		p[4] = math.sin(block*scl) # slightly cheating here
		p[5] = math.cos(block*scl)
		p[6] = i/9
		p[7] = j/9
		return p

	def encodeBoard(self, cursPos, board, guess, notes): 
		x = np.zeros((self.num_tokens, self.world_dim))
		
		# first token is the cursor (redundant -- might not be needed?)
		x[0, 0] = 1 # indicate this is the cursor token
		x[0, 1+9*3:] = self.encodePos(cursPos[0], cursPos[1])

		#encode the board state
		for i in range(9): 
			for j in range(9): 
				k = 1 + i*9 + j # token number
				if i == cursPos[0] and j == cursPos[1]: 
					x[k,0] = -1.0 # cursor on this square
				m = math.floor(board[i][j]) # works b/c 1 indexed.
				if m > 0: 
					x[k, m] = 1.0
				m = math.floor(guess[i][j]) # also ok: 1-indexed.
				if m > 0: 
					x[k, m+9] = 1.0
				x[k, 1+9*2:1+9*3] = notes[i,j,:]
				x[k,1+9*3:] = self.encodePos(i, j)
		return x
		
	def decodeBoardCursPos(self, board): 
		board_pos = board[1:, 1+9*3:-2]
		cursPos = board[0, 1+9*3:-2]
		# cursPos = cursPos.unsqueeze(0).expand(81, -1)
		cursPos = np.expand_dims(cursPos, axis=0)
		cursPos = np.repeat(cursPos, 81, axis=0)
		e = np.sum((cursPos - board_pos)**2, axis=1)
		indx = np.argmin(e).item()
		i = indx // 9
		j = indx % 9
		ii = board[0, -2]
		jj = board[0, -1]
		return [i,j,ii,jj]
		
	def decodeBoard(self, board, num,act,rin): 
		# input must be numpy ndarray
		# first the clues & (board) cursor position.
		clues = np.zeros((9,9))
		for i in range(9): 
			for j in range(9): 
				k = 1 + i*9 + j # token number
				block = i // 3 + j // 3
				attrs = []
				if board[k,0] < -0.15: 
					color = "blue"
					attrs = ["bold"]
				else:
					color = "black" if block % 2 == 0 else "red"
				v = np.argmax(board[k,:10])
				if board[k,v] <= 0.5:
					v = 0
				else:
					v = v.item()
				clues[i,j] = v
				if(len(attrs) > 0): 
					print(colored(v, color, attrs=attrs), end=" ")
				else: 
					print(colored(v, color), end=" ")
			print()
		# the encoded cursor position
		ci,cj,ii,jj = self.decodeBoardCursPos(board)
		print(f'decoded curs pos {ci},{cj} (sincos) / {ii},{jj} (linear)')
		print(f'decoded action {act} num {num}')
		
		snotes = np.zeros(9)
		sguess = np.zeros(9)
		# decode the notes, too. 
		for i in range(9): 
			for j in range(9): 
				k = 1 + i*9 + j # token number
				guess = board[k, 1+9*1:1+9*2]
				notes = board[k, 1+9*2:1+9*3]
				if i == ci and j == cj: 
					sclue = clues[i,j]
					snotes = notes
					sguess = guess
				# if np.sum(notes) > 0.5: 
				# 	print(f'notes[{i},{j}]:', end = '')
				# 	for l in range(9): 
				# 		e = notes[l]
				# 		if e > 0.0: 
				# 			print(f'{l+1}', end=',')
				# 	print(' ')
		valid = True
		for i in range(9): 
			for j in range(9): 
				if num == clues[i,j]: 
					if i == ci: 
						valid = False
					if j == cj:
						valid = False
					if i // 3 == ci // 3 and j // 3 == cj // 3: 
						valid = False
		# DEBUG: manually predict the reward, to make sure it's possible.
		reward = -0.05
		if act == 4: 
			# if snotes[num-1] > 0.5 and sclue < 1: 
			if valid and sclue < 0.5: 
				reward = 1.0
			else:
				reward = -1.0
		if act == 5: # does not depend on num.
			if np.sum(sguess) > 0.5: 
				reward = 0.0
			else: 
				reward = -0.25
		if act == 6: 
			if snotes[num-1] > 0.5 and sclue < 1: 
				reward = -0.25
		if act == 7: 
			if snotes[num-1] < 0.5: 
				reward = -0.25
		if abs(reward - rin) > 0.001:
			pdb.set_trace()
		return reward
	
	def forward(self, board_enc, latents): 
		# note: spatially, the number of latents = number of actions
		batch_size = board_enc.shape[0]
		lslow = self.latent_slow.unsqueeze(0).expand(batch_size, -1, -1)
		latents = th.cat((lslow, latents), 2)
		x = th.cat((board_enc, latents), 1)
		x = self.gelu(self.world_to_xfrmr(x))
		y = self.xfrmr(x)
		pdb.set_trace()
		nt = self.num_tokens
		new_board = self.xfrmr_to_world(y[:,0:nt, :]) # including cursor
		action = self.xfrmr_to_action(y[:,nt:,:])
		# for softmax, have to allow for "no action"=[0] and "no number"=[-1]
		action = th.cat( 
			(self.softmax(action[:,:,0:10]), self.softmax(action[:,:,10:])), 2)
		
		# given the board and suggested action, predict the reward. 
		# aslow = self.action_slow.unsqueeze(0).expand(batch_size, -1, -1)
		# action_l = th.cat((aslow, action), 2)
		# w = self.gelu(self.world_to_xfrmr(board_enc))
		# x = th.cat((w, y[:,nt:,:]), 1) # token dim
		# z = self.critic(x)
		reward = self.critic_to_reward(y[:,nt:,:]) # one reward for each action
		return new_board, action, reward
		
	def backLatent(self, board_enc, new_board, actions, reportFun): 
		# in supervised learning need to derive latent based on 
		# action, reward, and board state. 
		# yes, this is rather circular... 
		batch_size = board_enc.shape[0]
		latents = th.randn(batch_size, self.latent_cnt, self.world_dim // 2, requires_grad = True, device = board_enc.device) 
		for i in range(5):
			self.zero_grad()
			wp, ap, rp = self.forward(board_enc, latents / 10.0)
			# err = th.sum((new_board - wp)**2)*0.05 + \
			err = th.sum((actions - ap)**2)
				# no reward here!  latents only predict actions -- network needs to independently evaluate reward. 
				# otherwise we get short-circuit learning.
			err.backward()
			with th.no_grad():
				latents -= latents.grad * 1.0 # ??
				latents -= latents * 0.2 # weight decay
				if i == 4: 
					# pdb.set_trace()
					st = th.std(latents.grad)
					mn = th.mean(latents.grad)
					gf = (latents.grad - mn) / st # assume normal, zscore
					msk = th.abs(gf) < 0.02
					latents += (th.randn_like(latents) * msk) * err / 400 # must be last step?
					reportFun(board_enc, new_board, actions, wp, ap, rp, latents)
				scl = th.clip(th.std(latents, axis=-1), 1.0, 1e6) # std world dim
				# otherwise get interactions between tokens
				scl = scl.unsqueeze(-1).expand_as(latents)
				latents /= scl
			if False: 
				print("backLatent std,err:", th.std(latents).detach().cpu().item(), err.detach().cpu().item())
		# z-score the latents so we can draw from the same distro at opt time.  
		# print('----')
		latents = latents.detach() / 10.0
		s = th.clip(th.std(latents), 1.0, 1e6)
		latents = latents / s
		return latents
		
	def backLatentBoard(self, board_enc, new_board): 
		# infer the latents (and hence the actions and rewards) from beginning state and end state. 
		batch_size = board_enc.shape[0]
		latents = th.randn(batch_size, self.latent_cnt, self.world_dim // 2, requires_grad = True, device = board_enc.device)
		for i in range(10):
			self.zero_grad()
			wp, ap, rp = self.forward(board_enc, latents/10.0)
			err = th.sum((new_board - wp)**2)*0.05
			err.backward()
			with th.no_grad():
				latents -= latents.grad * 3.0 # ??
				latents -= latents * 0.06 # weight decay
				s = th.clip(th.std(latents), 1.0, 1e6)
				latents /= s
			if False: 
				print("backLatentBoard std,err:", th.std(latents).detach().cpu().item(), err.detach().cpu().item())
		# z-score the latents so we can draw from the same distro at opt time.  
		print('----')
		latents = latents.detach() / 10.0
		s = th.clip(th.std(latents), 1.0, 1e6)
		latents = latents / s
		return latents, ap, rp
		
	def backLatentReward(self, board_enc, reportFun): 
		# infer the latents (and hence the actions and rewards) from only beginning state.
		# assumes targeting a +1 reward.  FIXME
		batch_size = board_enc.shape[0]
		latents = th.randn(batch_size, self.latent_cnt, self.world_dim // 2, requires_grad = True, device = board_enc.device) 
		# latents.retain_grad()
		rewards = th.ones(batch_size, self.latent_cnt, 2, device=board_enc.device)
		for i in range(20):
			self.zero_grad()
			wp, ap, rp = self.forward(board_enc, latents/10.0)
			err = (1.0 - th.max(rp[:,:,0]))**2 + th.min(rp[:,:,0])**2 # target +1 cumulative reward
			err.backward()
			with th.no_grad():
				if False: 
					reportFun(board_enc, th.zeros_like(board_enc), th.zeros_like(ap), wp, ap, rp, latents)
					# pdb.set_trace()
				latents -= latents.grad * 2.0 # might bounce around a bit: limited supervision.
				latents -= latents * 0.06 # weight decay
				scl = th.clip(th.std(latents, axis=-1), 1.0, 1e6) # std world dim
				# otherwise get spurrious interactions between tokens
				latents /= s
			if False: 
				print("backLatentReward std,err:", th.std(latents).detach().cpu().item(), err.detach().cpu().item())
			del err # remove the computational graph..
		# z-score the latents so we can draw from the same distro at opt time.
		latents = latents.detach() / 10.0
		s = th.clip(th.std(latents), 1.0, 1e6)
		latents = latents / s
		return latents, wp, ap, rp
		
	def load_checkpoint(self, path:str=None):
		if path is None:
			path = "checkpoints/racoonizer.pth"
		self.load_state_dict(th.load(path))
   
	def save_checkpoint(self, path:str=None):
		if path is None:
			path = "checkpoints/racoonizer.pth"
		th.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")

	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params/1e6}M")
