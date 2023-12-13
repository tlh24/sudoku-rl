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

class STNFunction(torch.autograd.Function):
	# see https://www.kaggle.com/code/peggy1502/learning-pytorch-2-new-autograd-functions/notebook
	# and https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
	@staticmethod
	def forward(ctx, u, std, activ):
		# don't add noise to active tensors
		ushape = u.shape
		ac = torch.exp(-5.0*activ)
		s = torch.sum(ac, 2) # along model dim
		ac[:,:,0] = s*99 # probability of adding a new unit
		ac = torch.reshape(ac, (ushape[0]*ushape[1], ushape[2]))
		r = torch.multinomial(ac, 1)
		u = torch.reshape(u, (ushape[0]*ushape[1], ushape[2]))
		u[:,r] = u[:,r] + (std * (r > 0))
		u = torch.reshape(u, ushape)
		# ctx.save_for_backward(r)
		return u

	@staticmethod
	def backward(ctx, grad_output):
		# r2, = ctx.saved_tensors
		return grad_output, None, None # grad for: input, std, activ
		# return F.hardtanh(grad_output * x), None, None # clip gradients?
		# note: no need to gate the noise.
		# if the error is zero, grad_output will be zero as well.
		
class StraightThroughNormal(nn.Module):
	def __init__(self):
		super(StraightThroughNormal, self).__init__()
		self.activ = torch.randn(1)
		self.activ.requires_grad_(False)
		# self.register_buffer('activ', torch.randn(1))
		# don't do this - gradient calc leaks memory.

	def forward(self, x, std):
		if self.activ.shape != x.shape: 
			self.activ = torch.zeros_like(x)
			self.activ.requires_grad_(False)
		self.activ = 0.97 * self.activ + 0.03 * torch.abs(x.detach()) # or x^2 ?
		x = STNFunction.apply(x, std, self.activ)
		return x

class ResidualAttentionBlock(nn.Module): 
	def __init__(self, d_model: int, n_head: int, init_zeros:bool):
		super().__init__()
		# need to be able to both infer attention (= head activation) from Q,K activity, as well as set it from straight search.  
		# I'm not totally sure of how to do this. 
		self.n_head = n_head
		self.d_model = d_model
		self.init_zeros = init_zeros
		# self.ln_1 = LayerNorm(d_model) # unused
		# self.ln_2 = LayerNorm(d_model) # unused
		self.wqkv = nn.Linear(d_model, 3*d_model, bias=False)
		self.soft = torch.nn.Softmax(dim=2)
		self.fanout = nn.Linear(d_model, d_model * 3)
		self.fanout_stn = StraightThroughNormal()
		self.gelu = QuickGELU()
		self.fanin = nn.Linear(d_model * 3, d_model)
		self.fanin_stn = StraightThroughNormal()
		if init_zeros: 
			torch.nn.init.zeros_(self.wqkv.weight)
			torch.nn.init.zeros_(self.fanout.weight)
			torch.nn.init.zeros_(self.fanin.weight)
			torch.nn.init.zeros_(self.fanout.bias)
			torch.nn.init.zeros_(self.fanin.bias)
		
	def attention(self, x:torch.Tensor, std:float): 
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		d_head = self.d_model // self.n_head
		y = self.wqkv(x)
		y = torch.reshape(y, (batch_size, ntok, self.n_head, d_head*3) )
		q,k,v = torch.split(y, d_head, dim=-1)
		a = torch.einsum('bthd,bshd -> btsh', q, k) / math.sqrt(d_head)
		a = self.soft(a)
		b = torch.einsum('btsh,bshd -> bthd', a, v)
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		return b # residual sum later.

	def forward(self, x:torch.Tensor):
		if self.init_zeros:
			std = 0.01 / 6.0 # low sensitivity here
		else:
			std = 0.0
		y = self.attention(x, std) # should this be x or y?
		# y = self.fanout_stn(self.fanout(y), std)
		y = self.fanout(y)
		y = self.gelu(y)
		# y = self.fanin_stn(self.fanin(y), std)
		y = self.fanin(y)
		# y = self.gelu(y) # ??
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

	def forward(self, x:torch.Tensor):
		for i in range(self.repeat): 
			# one-hot encode the layer position on all tokens. 
			x[:,:,self.d_model - self.repeat : self.d_model] = 0.0
			x[:,:,self.d_model - i - 1] = 1.0
			x = self.layer1(x)
			x = self.layer2(x)
		return x
