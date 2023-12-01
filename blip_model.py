import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pdb

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module): 
	def __init__(self, d_model: int, n_head: int, init_zeros:bool):
		super().__init__()
		# need to be able to both infer attention (= head activation) from Q,K activity, as well as set it from straight search.  
		# I'm not totally sure of how to do this. 
		self.n_head = n_head
		self.d_model = d_model
		self.ln_1 = LayerNorm(d_model)
		self.ln_2 = LayerNorm(d_model)
		self.wqkv = nn.Linear(d_model, 3*d_model, bias=False)
		self.soft = torch.nn.Softmax(dim=2)
		self.fanout = nn.Linear(d_model, d_model * 3)
		self.gelu = QuickGELU()
		self.fanin = nn.Linear(d_model * 3, d_model)
		if init_zeros: 
			torch.nn.init.zeros_(self.wqkv) # bias starts at zero by default
			torch.nn.init.zeros_(fanout.weight)
			torch.nn.init.zeros_(fanin.weight)
		
	def attention(self, x:torch.Tensor, std:float): 
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		d_head = self.d_model // self.n_head
		y = torch.reshape(self.wqkv(x), (batch_size, ntok, self.n_head, d_head*3) )
		q,k,v = torch.split(y, d_head, dim=-1)
		a = torch.einsum('bthd,bshd -> btsh', q, k) / math.sqrt(d_head)
		if std > 0.0:
			a = a + torch.randn_like(a) * std
		a = self.soft(a)
		b = torch.einsum('btsh,bshd -> bthd', a, v)
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		return x + b

	def forward(self, x:torch.Tensor, std:float): 
		x = x + self.attention(x, std)
		y = self.fanout(x)
		if std > 0.0: 
			y = y + torch.randn_like(y) * std
		y = self.gelu(y)
		y = self.fanin(y)
		return x + y
		
		
class Transformer(nn.Module): 
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int, init_zeros:bool):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.layer1 = ResidualAttentionBlock(d_model, n_head, init_zeros)
		self.layer2 = ResidualAttentionBlock(d_model, n_head, init_zeros)
		# self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model, n_head, init_zeros) for _ in range(layers)])

	def forward(self, x:torch.Tensor, std:float):
		for i in range(self.repeat): 
			# one-hot encode the layer position on all tokens. 
			x[:,:,self.d_model - self.repeat : self.d_model] = 0.0
			x[:,:,self.d_model - i - 1] = 1.0
			x = self.layer1(x, std=std)
			x = self.layer2(x, std=std)
		return x
