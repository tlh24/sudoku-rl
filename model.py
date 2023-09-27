import math
import torch as th
from torch import nn
import clip_model
import pdb

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
		
		self.xfrmr = clip_model.Transformer(
			width = xfrmr_width, 
			layers = 6, 
			heads = 4, 
			attn_mask = None)
		
		self.xfrmr_to_world = nn.Linear(xfrmr_width, world_dim)
		self.xfrmr_to_action = nn.Linear(xfrmr_width, action_dim)
		self.softmax = nn.Softmax(dim = 2)
		self.xfrmr_to_reward = nn.Linear(xfrmr_width, 2) 
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
	
	def forward(self, board_enc, latents): 
		# note: spatially, the number of latents = number of actions
		batch_size = board_enc.shape[0]
		lslow = self.latent_slow.unsqueeze(0).expand(batch_size, -1, -1)
		latents = th.cat((lslow, latents), 2)
		x = th.cat((board_enc, latents), 1)
		x = self.gelu(self.world_to_xfrmr(x))
		y = self.xfrmr(x)
		new_board = self.xfrmr_to_world(y[:,0:82, :]) # including cursor
		action = self.xfrmr_to_action(y[:,82:,:])
		# for softmax, have to allow for "no action"=[0] and "no number"=[-1]
		action = th.cat( 
			(self.softmax(action[:,:,0:10]), self.softmax(action[:,:,10:])), 2)
		reward = self.xfrmr_to_reward(y[:,82:,:])
		return new_board, action, reward
		
	def backLatent(self, board_enc, new_board, actions, rewards): 
		# in supervised learning need to derive latent based on 
		# action, reward, and board state. 
		# yes, this is rather circular... 
		batch_size = board_enc.shape[0]
		latents = th.randn(batch_size, self.latent_cnt, self.world_dim // 2, requires_grad = True, device = board_enc.device) 
		for i in range(5):
			self.zero_grad()
			wp, ap, rp = self.forward(board_enc, latents / 10.0)
			err = th.sum((new_board - wp)**2)*0.05 + \
					th.sum((actions - ap)**2) + \
					th.sum((rewards - rp)**2) 
			err.backward()
			with th.no_grad():
				latents -= latents.grad * 0.15 # ??
				latents -= latents * 0.03 # weight decay
			if i == 4: 
				print("backLatent std,err:", th.std(latents).detach().cpu().item(), err.detach().cpu().item())
		# z-score the latents so we can draw from the same distro at opt time.  
		latents = latents.detach() / 10.0
		s = th.clip(th.std(latents), 1.0, 1e6)
		latents = latents / s
		return latents
		
	def backLatentBoard(self, board_enc, new_board): 
		# infer the latents (and hence the actions and rewards) from beginning state and end state. 
		batch_size = board_enc.shape[0]
		latents = th.randn(batch_size, self.latent_cnt, self.world_dim // 2, requires_grad = True, device = board_enc.device)
		for i in range(5):
			self.zero_grad()
			wp, ap, rp = self.forward(board_enc, latents/10.0)
			err = th.sum((new_board - wp)**2)*0.05
			err.backward()
			with th.no_grad():
				latents -= latents.grad * 0.15 # ??
				latents -= latents * 0.03 # weight decay
			if True: 
				print("backLatentBoard std,err:", th.std(latents).detach().cpu().item(), err.detach().cpu().item())
		# z-score the latents so we can draw from the same distro at opt time.  
		latents = latents.detach() / 10.0
		s = th.clip(th.std(latents), 1.0, 1e6)
		latents = latents / s
		return latents, ap, rp
		
	def backLatentReward(self, board_enc): 
		# infer the latents (and hence the actions and rewards) from only beginning state.
		# assumes targeting a +1 reward.  FIXME
		batch_size = board_enc.shape[0]
		latents = th.randn(batch_size, self.latent_cnt, self.world_dim // 2, requires_grad = True, device = board_enc.device) 
		# latents.retain_grad()
		rewards = th.ones(batch_size, self.latent_cnt, 2, device=board_enc.device)
		for i in range(20):
			self.zero_grad()
			wp, ap, rp = self.forward(board_enc, latents/10.0)
			err = th.sum((rewards[:,:,1] - rp[:,:,1])**2)
			err.backward()
			with th.no_grad():
				latents -= latents.grad * 1.0 # might bounce around a bit: only 1-bit supervision.
				latents -= latents * 0.02 # weight decay
			if False: 
				print("backLatentReward std,err:", th.std(latents).detach().cpu().item(), err.detach().cpu().item())
			del err # remove the computational graph..
		# z-score the latents so we can draw from the same distro at opt time.
		latents = latents.detach() / 10.0
		s = th.clip(th.std(latents), 1.0, 1e6)
		latents = latents / s
		return latents, ap, rp

	def print_n_params(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params/1e6}M")
