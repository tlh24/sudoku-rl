import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchlayers as tl
import l1attn_sparse_cuda
import l1attn_cuda
# from flash_attn import flash_attn_func # half only
import pdb
import matplotlib.pyplot as plt
from constants import g_zeroinit, g_l1atten, SuN, g_dtype, token_cnt

class LayerNorm(nn.LayerNorm):
	"""Subclass torch's LayerNorm to handle fp16."""
	def forward(self, x: torch.Tensor):
		orig_type = x.dtype
		ret = super().forward(x.type(torch.float32))
		return ret.type(orig_type)

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)
	  
class LinearM(nn.Module): 
	# with the bias merged -- used for PSGD optimizer.
	def __init__(self, indim:int, outdim:int, initzeros:bool): 
		super(LinearM, self).__init__()
		scl = 0.02 / math.sqrt(2*9)
		if initzeros: 
			self.w = torch.nn.Parameter( scl * torch.ones(outdim,indim+1,dtype=g_dtype))
		else:
			self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim+1,dtype=g_dtype))
		with torch.no_grad(): 
			self.w[:,-1] = 0.0 # bias starts at 0
		
	def forward(self, x):
		return torch.einsum('oi,bhi -> bho', self.w[:,:-1], x) + self.w[:,-1]
		
class LinearNobias(nn.Module): 
	# with the bias merged.
	def __init__(self, indim:int, outdim:int, initzeros:bool): 
		super(LinearNobias, self).__init__()
		scl = 0.02 / math.sqrt(2 * 9)
		if initzeros: 
			scl = 0.0
		self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim,dtype=g_dtype) )
		
	def forward(self, x):
		return torch.einsum('oi,bhi -> bho', self.w[:, :-1], x)

class STNFunction(torch.autograd.Function):
	# see https://www.kaggle.com/code/peggy1502/learning-pytorch-2-new-autograd-functions/notebook
	# and https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
	@staticmethod
	def forward(ctx, input, std, activ):
		batch_size = input.shape[0]
		ac = torch.squeeze(activ)
		ac = torch.exp(-5.0*ac)
		s = torch.sum(ac)
		ac[0] = s*999 # controls the probability of adding a new unit
		# unit zero is hence never enabled.
		# this causes scaling problems... meh.
		r = torch.multinomial(ac, batch_size, replacement=True) # sample 1 row to activate, based on the probability distribution 'ac'.
		i = torch.arange(batch_size)
		x = input
		x[i,0,r] = x[i,0,r] + (std * (r > 0))
		return x

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output, None, None # grad for: input, std, activ
		
class StraightThroughNormal(nn.Module):
	def __init__(self,n):
		super(StraightThroughNormal, self).__init__()
		self.register_buffer('activ', torch.zeros(1,n))

	def forward(self, x, std):
		self.activ = 0.97 * self.activ + 0.03 * torch.mean(torch.abs(x), dim=0) # or x^2 ?
		x = STNFunction.apply(x, std, self.activ.detach())
		return x
		
# class STAFunction(torch.autograd.Function): 
# 	@staticmethod
# 	def forward(ctx, a, activ):
# 		# activ is head dimensioned
# 		ashape = a.shape
# 		batch_size = ashape[0]
# 		nheads = ashape[-1]
# 		ntok = ashape[1]
# 		ac = torch.exp(-5.0 * activ)
# 		s = torch.sum(ac)
# 		# add a 'none' selection at the end
# 		ac = torch.cat((ac, s.expand(1)*99*batch_size))
# 		ac = ac.unsqueeze(0).repeat(batch_size, 1)
# 		r = torch.multinomial(ac, 1)
# 		g = torch.zeros((batch_size, nheads+1), device=a.device)
# 		g[:,r] = 3.0
# 		g = g[:,0:-1]
# 		if torch.sum(g) > 0: 
# 			print("!!firing random attention!!")
# 		r = g.unsqueeze(1).unsqueeze(2).repeat((1,ntok,ntok,1))
# 		y = torch.nn.functional.relu(torch.randn_like(a) - 3.0) * r
# 		return a + y
# 		
# 	def backward(ctx, grad_output): 
# 		return grad_output, None
# 		
# class StraightThroughAttention(nn.Module):
# 	def __init__(self):
# 		super(StraightThroughAttention, self).__init__()
# 		self.activ = torch.randn(1)
# 		self.activ.requires_grad_(False)
# 
# 	def forward(self, a):
# 		if self.activ.shape != a.shape[-1]: 
# 			self.activ = torch.zeros(a.shape[-1], device=a.device)
# 			self.activ.requires_grad_(False)
# 		s = torch.sum(torch.abs(a.detach()), (0,1,2)) # batch, tok, tok
# 		self.activ = 0.97 * self.activ + 0.03 * s # or x^2 ?
# 		a = STAFunction.apply(a, self.activ)
# 		return a

class ResidualAttentionBlock(nn.Module): 
	def __init__(self, d_model: int, n_head: int, init_zeros:bool):
		super().__init__()
		# need to be able to both infer attention (= head activation) from Q,K activity, as well as set it from straight search.  
		# I'm not totally sure of how to do this. 
		self.n_head = n_head
		self.d_model = d_model
		self.init_zeros = init_zeros
		# what if we init the query with 0.005 * diag(d_model)? 
		# then you rely only on the value & output MLP (both randomly initialized) to push differentiation of the heads. 
		self.wq = LinearM(d_model, n_head*d_model, init_zeros) # constant init works fine, just a bit slower. 
		self.wv = LinearM(d_model, n_head*d_model, init_zeros)
		#self.wqv = LinearM(d_model, n_head*2*d_model, init_zeros) 
		self.wk = torch.nn.Parameter( 0.005 * torch.ones(n_head, d_model) )
		# self.wk = LinearNobias(d_model, n_head, False) # not zeroinit
		# self.bk = LinearNobias(d_model, n_head, True) # zeroinit
		# self.wk = LinearNobias(d_model, n_head*d_model, True) # full rank key calc
			# wk is just a weighting, not a full matrix 
			# to avoid double permutation invariance.
			# starts out at zero = head gated off.
		self.head_enabled = [not init_zeros for _ in range(n_head)]
		self.head_enabled[-1] = True # all-to-all always on.
		self.l1a_s = l1attn_sparse_cuda.L1AttnSparse()
		self.l1a_f = l1attn_cuda.L1Attn()
		self.soft = torch.nn.Softmax(dim=2) # unused with L1 attn
		self.fanout = LinearM(d_model, d_model * 1, False) # non-zero init
		# self.fanout = torch.nn.Parameter( torch.ones(2, d_model) )
		# self.fanout[1,:] = 0.0
		#self.fanout_stn = StraightThroughNormal() # try this again?
		self.gelu = QuickGELU()
		# self.fanin = nn.Linear(d_model * 3, d_model)
		# self.fanin_stn = StraightThroughNormal()
		
		
	def attention(self, x:torch.Tensor, hcoo:list, n:int, layer:int, pas:int, record=list):
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		width = x.shape[2]
		
		if pas == 0 and self.init_zeros: 
			schedule = 2000 # slower for SGD
			init_head = n // schedule
			if n % schedule == layer*(schedule//2) and init_head < self.n_head and (not self.head_enabled[init_head]): # only 2 layers! 
				with torch.no_grad(): 
					w = self.wk
					w[init_head, :] = 1.0 # no division -- just a gate! 
					# indx = init_head*d_head
					# w[indx:indx+d_head, :] = torch.randn(d_head, d_head) / math.sqrt(d_head) # full rank key calc
					self.wk.copy_( w )
					self.head_enabled[init_head] = True
					print(f"initialized head {init_head} of layer {layer}")
		
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		
		# y = self.wqv(x)
		# if record is not None: 
		# 	record.append( y ) # with grad!
		# y = torch.reshape(y, (batch_size, ntok, self.n_head, d_head*2) )
		# q,v = torch.split(y, d_head, dim=-1) 
			# q-v hence becomes second to last dim
		q = self.wq(x)
		q = torch.reshape(q, (batch_size, ntok, self.n_head, d_head))
		v = self.wv(x)
		v = torch.reshape(v, (batch_size, ntok, self.n_head, d_head))
		
		
		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		
		gk = self.wk.unsqueeze(0).unsqueeze(0).expand([batch_size,ntok,-1,-1])
		# bk = self.bk.w.unsqueeze(0).unsqueeze(0).expand([batch_size,ntok,-1,-1])
		k = k * gk # + bk # with bias to allow for centering.
		
		# full-rank key calculation (worse!)
		# kw = self.wk.w.reshape([self.n_head, self.d_model, -1])
		# k = torch.einsum("btd,hde -> bthe", x, kw)
	
		# gate the value by head_enabled. 
		# during sampling, all heads enabled, so no need to save. 
		gv = torch.tensor(self.head_enabled, dtype=g_dtype, device=v.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand([batch_size,ntok,-1,d_head])
		v = v * gv  
		
		if g_l1atten: 
			# cycle through the coo vectors.  
			if layer % 4 == 3: 
				if True:
					# extract all global / all-to-all tokens
					# really could do this with pure sparse attn.. will have to compare. 
					a2a = hcoo[3]
					a2len = a2a.shape[0]
					q = q[:,a2a,:,:]
					k = k[:,a2a,:,:]
					v = v[:,a2a,:,:]
					# pad out to BLKSIZ tokens (for CUDA kernel).
					padn = ((a2len + 15) // 16) * 16 - a2len
					assert(padn > 0) # for noop
					qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
					kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
					a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
					a = a[:, :a2len+1, :a2len, :]
					a[:, a2len, :,:] = 0.0
					# add in e^0=1 as a 'noop' option
					# (hence max attention is 0.5, not 1)
					# output is b,src,dst,heads
					a = F.softmax(a, 1) # see l1attn.py -- sm over src
					a = a[:, :a2len, :a2len, :] # remove noop
					bb = torch.einsum('bsdh, bshw -> bdhw', a, v)
					# scatter to original sites
					b = torch.zeros(batch_size, ntok, n_head, width, device=v.device)
					indx = torch.arange(0, a2len, device=v.device)
					b[:,a2a,:,:] = bb[:,indx,:,:]
				# else: 
				# 	# this is dot-product attention
				# 	a = torch.einsum('bthd,bshd -> btsh', q, k) / math.sqrt(d_head)
				# 	a = self.soft(a)
				# 	b = torch.einsum('btsh,bshd -> bthd', a, v)
				# else: 
				# 	# flashAttention only supports float16
				# 	b = flash_attn_func(q.half(), k.half(), v.half())
				# 	b = b.float()
			else: 
				coo,dst_mxlen = hcoo[layer%4] 
				use_softmax = True 
				b = self.l1a_s(v,q,k,coo,dst_mxlen,use_softmax) 
			ap = torch.zeros(ntok, ntok, n_head) # dummy.
		else: 
			a = torch.einsum('bthd,bshd -> btsh', q, k) / math.sqrt(d_head)
			a = self.soft(a)
			# a = a * msk 
			b = torch.einsum('btsh,bshd -> bthd', a, v) # regular attention
			# ap = (a[0,:,:,:] - 1.0 + msk[0,:,:,:]).squeeze().detach().cpu()
			ap = (a[0,:,:,:]).squeeze().detach().cpu()
		
		# multiply b by a symmetry-breaking mask
		# d_model = self.d_model
		# msk = torch.ones(self.n_head, d_model) * 1e-3
		# msk[:, 0:21] = 1
		# for i in range(self.n_head):
		# 	msk[i, d_model-(i+1)*3:d_model-i*3] = 1
		# msk = msk.to(b.device)
		# msk = msk.unsqueeze(0).unsqueeze(1).expand([batch_size,ntok,-1,-1])
		# b = torch.sum(b*msk, dim=2) # sum along the heads
		
		b = torch.sum(b, dim=2) # sum along the heads
		
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		 
		return b,ap # residual sum later.

	def forward(self, x:torch.Tensor, hcoo:list, n:int, layer:int, pas:int, record=list):
		y,ap = self.attention(x,hcoo,n,layer,pas,record)
		if record is not None: 
			record.append( y )
		# y = self.ln_1(y) # stabilize learning? 
		# y = self.fanout_stn(self.fanout(y), 0.01)
		# y = self.fanout(y)
		# y = self.gelu(y+SuN/2.0)-(SuN/2.0) # this nonlinearity is essential
		y = self.gelu(y)
		# y = self.fanout[0,:] * y + self.fanout.w[1,:]
		y = self.fanout(y) # allow sign inversions & mixing; no dim change
		# y = self.gelu(y) # this destroys performance! 
		# y = self.fanout_stn(y, 0.01)
		# y = self.fanin(y)
		# y = self.gelu(y) # ??
		# y = self.ln_2(y) # stabilize	 learning? 
		return x + y, ap, None
		
	def allHeadsOn(self): 
		self.head_enabled = [True for _ in range(self.n_head)]
		
		
class Transformer(nn.Module): 
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int, init_zeros:bool):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		#self.layer1 = ResidualAttentionBlock(d_model, n_head, init_zeros)
		self.resblocks = nn.ModuleList([ResidualAttentionBlock(d_model, n_head, init_zeros) for _ in range(layers)])

	def forward(self, x:torch.Tensor, hcoo:list, n:int, record:list):
		for i in range(self.repeat): 
			for j, layer in enumerate(self.resblocks):
				# linearly encode the repeat position on all tokens. 
				x[:,:,0] = i*2
				x,a1,w1 = layer(x,hcoo,n,j,i,record)
				if j == 1: 
					a2 = a1
					w2 = w1
		return x, w1, w2 # dumb way to do this

	def allHeadsOn(self): 
		self.layer1.allHeadsOn()
		self.layer2.allHeadsOn()

	def backAction(self, x, msk, y, record, denoisenet, denoisestd, temp, record_true, doplot=False): 
		# x needs to have grad on! 
		if doplot: 
			fig, axs = plt.subplots(5, 1, figsize=(35,20))
		
		losses = []
		# accumulate gradients from the denoised hidden states. 
		for j,h in enumerate(record):
			if j > -1: # ignore action. (saved in gracoonizer, not here)
				h = torch.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]))
				with torch.no_grad(): 
					net = denoisenet[j]
					std = denoisestd[j]
					z = torch.randn_like(h) * std * math.sqrt(temp) * 0.1
					t = torch.ones(h.shape[0], device=x.device) * temp
					hdn = net.forward(h+z,t)
				loss = torch.sum((h - hdn)**2) / np.prod(h.shape)
				loss.backward(retain_graph=True)
				losses.append(loss.detach().cpu().item())
			else: 
				losses.append(0.0)
			
			if doplot: 
				if j < 5 and j > -1: 
					ht = record_true[j]
					ht = torch.reshape(ht, (ht.shape[0], ht.shape[1]*ht.shape[2]))
					axs[j].plot(ht[0,:].detach().cpu().numpy().T, 'k')
					axs[j].plot(h[0,:].detach().cpu().numpy().T, 'b')
					axs[j].plot(hdn[0,:].detach().cpu().numpy().T, 'r')
					axs[j].set_title(f'{j} k=truth; b=estimate; r=denoised')
			
		plt.show()
		return losses
