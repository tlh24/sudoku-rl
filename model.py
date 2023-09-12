import torch as th
from torch import nn
import torch.cuda.amp
from ctypes import *
import clip_model
from pathlib import Path
from typing import Union


class Recognizer(nn.Module):
	
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
		
		self.model = clip_model.Transformer(
			width = xfrmr_width, 
			layers = 4, 
			heads = 4, 
			attn_mask = None)
		
		model_to_action = nn.Linear(xfrmr_width, action_dim)
		model_to_reward = nn.Linear(xfrmr_width, 2) # immedate and infinite-horizon
		
		
	def encodePos(i, j): 
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
		x = th.zeros(1 + 81, world_dim)
		
		# encode the cursor token
		x[0, 0] = 1
		x[0, 1+9*3:] = encodePos(cursPos[0], cursPos[1])

		#encode the board state
		for i in range(9): 
			for j in range(9): 
				k = 1 + i*9 + j
				m = board[i][j] 
				if m > 0: 
					x[k, m] = 1.0
				m = guess[i][j]
				if m > 0: 
					x[k, m+9] = 1.0
				x[k, 1+9*2:1+9*3] = notes[i,j,:]
				x[k,1+9*3:] = encodePos(i, j)
		return x
		
	
	def forward(self, board_enc, latents): 
		x = th.cat((board_enc, latents), 0)
		
		y = model(x)
		action = model_to_action(y[0,:])
		reward = model_to_reward(y[0,:])
		
		
		# encode the image (we should only need to do this once??)
		q = th.zeros(6) # ! this will be parallelized !
		vx = self.vit(batch_a) # x is size [bs, v_ctx, 256] 
		q[0] = th.std(vx)
		vx = self.vit_to_prt(vx)
		q[1] = th.std(vx)
		# vx = gelu(vx) # ? needed ? 

		px = self.encoder(batch_p)
		q[2] = th.std(px)
		vxpx = th.cat((vx, px), dim = 1)
		q[3] = th.std(vxpx)
		# x = vxpx * mask
		x = self.prt(vxpx) # bs, v_ctx + p_ctx, prog_width
		q[4] = th.std(x)
		x = th.reshape(x, (-1,(self.v_ctx + self.p_ctx)*self.prog_width))
		# batch size will vary with dataparallel
		x = self.prt_to_edit(x)
		q[5] = th.std(x)
		# x = self.ln_post(x) # scale the inputs to softmax
		# x = self.gelu(x)
		# x = th.cat((self.tok_softmax(x[:,0:4]),
		# 		  self.tok_softmax(x[:,4:4+toklen]), 
		# 		  x[:,4+toklen:]), dim=1) -- this is for fourier position enc. 
		return x,q

	@staticmethod
	def build_attention_mask(v_ctx, p_ctx):
		# allow the model to attend to everything when predicting an edit
		# causalty is enforced by the editing process.
		# see ec31.py for a causal mask.
		ctx = v_ctx + p_ctx
		mask = th.ones(ctx, ctx)
		return mask
	
	def load_checkpoint(self, path: Union[Path, str]=None):
		if path is None:
			path = self.CHECKPOINT_SAVEPATH
			self.load_state_dict(th.load(path))
   
	def save_checkpoint(self, path: Union[Path, str]=None):
		if path is None:
			path = self.CHECKPOINT_SAVEPATH
		torch.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")
   
	
	def print_model_params(self): 
		print(self.prt_to_tok.weight[0,:])
		print(self.prt.resblocks[0].mlp[0].weight[0,:])
		print(self.vit_to_prt.weight[0,1:20])
		print(self.vit.transformer.resblocks[0].mlp[0].weight[0,1:20])
		print(self.vit.conv1.weight[0,:])
		# it would seem that all the model parameters are changing.
  
	def std_model_params(self): 
		q = th.zeros(5)
		q[0] = th.std(self.vit.conv1.weight)
		q[1] = th.std(self.vit.transformer.resblocks[0].mlp[0].weight)
		q[2] = th.std(self.vit_to_prt.weight)
		q[3] = th.std(self.prt.resblocks[0].mlp[0].weight)
		q[4] = th.std(self.prt_to_tok.weight)
		return q

	def print_n_params(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params/1e6}M")
