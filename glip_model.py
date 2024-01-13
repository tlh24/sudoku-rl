import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchlayers as tl
import l1attn
import pdb
import matplotlib.pyplot as plt
from constants import g_zeroinit, g_l1atten

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
		ac[:,:,0] = s*99 # zero is the null unit; sets probability of perturb
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
		
class STAFunction(torch.autograd.Function): 
	@staticmethod
	def forward(ctx, a, activ):
		# activ is head dimensioned
		ashape = a.shape
		batch_size = ashape[0]
		nheads = ashape[-1]
		ntok = ashape[1]
		ac = torch.exp(-5.0 * activ)
		s = torch.sum(ac)
		# add a 'none' selection at the end
		ac = torch.cat((ac, s.expand(1)*99*batch_size))
		ac = ac.unsqueeze(0).repeat(batch_size, 1)
		r = torch.multinomial(ac, 1)
		g = torch.zeros((batch_size, nheads+1), device=a.device)
		g[:,r] = 3.0
		g = g[:,0:-1]
		if torch.sum(g) > 0: 
			print("!!firing random attention!!")
		r = g.unsqueeze(1).unsqueeze(2).repeat((1,ntok,ntok,1))
		y = torch.nn.functional.relu(torch.randn_like(a) - 3.0) * r
		return a + y
		
	def backward(ctx, grad_output): 
		return grad_output, None
		
class StraightThroughAttention(nn.Module):
	def __init__(self):
		super(StraightThroughAttention, self).__init__()
		self.activ = torch.randn(1)
		self.activ.requires_grad_(False)

	def forward(self, a):
		if self.activ.shape != a.shape[-1]: 
			self.activ = torch.zeros(a.shape[-1], device=a.device)
			self.activ.requires_grad_(False)
		s = torch.sum(torch.abs(a.detach()), (0,1,2)) # batch, tok, tok
		self.activ = 0.97 * self.activ + 0.03 * s # or x^2 ?
		a = STAFunction.apply(a, self.activ)
		return a

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
		self.wqkv = nn.Linear(d_model, n_head*2*d_model) 
		# self.wqkv = tl.L1(nn.Linear(d_model, n_head*2*d_model), weight_decay = 5e-2) # seems to be slower??
		self.wk = nn.Linear(d_model, n_head, bias=False)
		# self.bv = nn.Linear(d_model, n_head, bias=False)
		self.head_enabled = [not g_zeroinit for _ in range(n_head)]
		self.head_enabled[-1] = True # all-to-all always on.
		self.l1a = l1attn.L1Attn()
		# self.sta = StraightThroughAttention()
		if g_l1atten: # axis 2 for regular attention, 3 for l1
			self.soft = torch.nn.Softmax(dim=3) 
		else: 
			self.soft = torch.nn.Softmax(dim=2)
		self.fanout = nn.Linear(d_model, d_model * 1)
		self.fanout_stn = StraightThroughNormal()
		self.gelu = QuickGELU()
		self.fanin = nn.Linear(d_model * 3, d_model)
		self.fanin_stn = StraightThroughNormal()
		if g_zeroinit:
			with torch.no_grad(): 
				w = torch.zeros_like(self.wqkv.weight)
				self.wqkv.weight.copy_(w)
				w = torch.zeros_like(self.wqkv.bias)
				self.wqkv.bias.copy_(w)
				# w = torch.eye(d_model, d_model)
				# self.fanout.weight.copy_(w)
				w = torch.zeros_like(self.wk.weight)
				w[:,-1] = 1.0 # all-to-all always on
				self.wk.weight.copy_(w)
		
	def attention(self, x:torch.Tensor, msk:torch.Tensor, n:int, layer:int, pas:int):
		if pas == 0 and g_zeroinit: 
			schedule = 600
			init_head = n // schedule
			if n % schedule == layer*(schedule//2) and init_head < self.n_head: # only 2 layers! 
				with torch.no_grad(): 
					w = self.wk.weight
					w[init_head, :] = 1.0 # no division -- just a gate! 
					self.wk.weight.copy_( w )
					self.head_enabled[init_head] = True
					print(f"initialized head {init_head}")
		
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		d_head = self.d_model
		
		y = self.wqkv(x)
		y = torch.reshape(y, (batch_size, ntok, self.n_head, d_head*2) )
		q,v = torch.split(y, d_head, dim=-1) 
			# q-v hence becomes second to last dim
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		gk = self.wk.weight.unsqueeze(0).unsqueeze(0).expand([batch_size,ntok,-1,-1])
		k = k * gk
		gv = torch.tensor(self.head_enabled, dtype=torch.float32, device=v.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand([batch_size,ntok,-1,d_head])
		v = v * gv  # bias term to the value (since there is no subsequent mlp layer)
		# q = torch.nn.functional.normalize(q, dim=3, eps=1e-2) # doesn't work!! 
		# v = torch.nn.functional.normalize(v, dim=3, eps=1e-2) 
		# a = torch.einsum('bthd,bshd -> btsh', q, k) / math.sqrt(d_head)
		if g_l1atten: 
			a = self.l1a(q,k) / math.sqrt(d_head) # output is bhts!
			a = self.soft(a) # v gets updated from the gradient, but q and k do not.
			# a = self.gelu(a)
			a = a * msk
			b = torch.einsum('bhts,bshd -> bthd', a, v) 
			ap = (a[0,:,:,:] - 1.0 + msk[0,:,:,:]).squeeze().detach().cpu() #
			ap = torch.permute(ap, (1,2,0)).contiguous() # for viewing.
		else: 
			a = torch.einsum('bthd,bshd -> btsh', q, k) / math.sqrt(d_head)
			a = self.soft(a) # v gets updated from the gradient, but q and k do not.
			# a = self.gelu(a) # need bias for this to work
			a = a * msk
			b = torch.einsum('btsh,bshd -> bthd', a, v) # regular attention
			ap = (a[0,:,:,:] - 1.0 + msk[0,:,:,:]).squeeze().detach().cpu() #
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		 
		return b,ap # residual sum later.

	def forward(self, x:torch.Tensor, msk:torch.Tensor, n:int, layer:int, pas:int):
		y,ap = self.attention(x,msk,n,layer,pas) 
		# y = self.ln_1(y) # stabilize learning? 
		# y = self.fanout_stn(self.fanout(y), std)
		# y = self.fanout(y)
		y = self.gelu(y+4.0)-4.0 # i think this nonlinearity is helpful
		y = self.fanout(y) # allow sign inversions.
		# y = self.fanin_stn(self.fanin(y), std)
		# y = self.fanin(y)
		# y = self.gelu(y) # ??
		# y = self.ln_2(y) # stabilize learning? 
		return x + y, ap, self.wqkv.weight.detach().cpu()
		
		
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

	def forward(self, x:torch.Tensor, msk:torch.Tensor, n:int):
		for i in range(self.repeat): 
			# # one-hot encode the layer position on all tokens. 
			# x[:,:,self.d_model - self.repeat : self.d_model] = 0.0
			# x[:,:,self.d_model - i - 1] = 1.0
			x,a1,w1 = self.layer1(x,msk,n,0,i)
			x,a2,w2 = self.layer2(x,msk,n,1,i)
			# x,a3,w3 = self.layer3(x,msk,n,1)
		return x, a1, a2, w1, w2
