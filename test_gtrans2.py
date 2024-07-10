# want to test the graph transformer on a few basic tasks
# e.g. is an element in the set, 
# what is the distance to match a given pattern.  

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from graph_transformer import LinearM, QuickGELU
import l1attn_cuda
import matplotlib.pyplot as plt
import pdb
import psgd 

npos = 10
ntok = npos + 4
width = 10
batch_size = 32

def genData(nn): 
	y = torch.zeros(nn, ntok, width)
	target = torch.zeros(nn)
	lin = torch.arange(0,npos)
	x = torch.zeros(npos, width)
	# binary encoding of the digits.  
	x[:,0] = lin % 2
	x[:,1] = (lin//2) % 2
	x[:,2] = (lin//4) % 2
	x[:,3] = (lin//8) % 2
	for n in range(nn): 
		# shuffle tokens
		indx = torch.randperm(npos)
		# print(indx)
		# indx = torch.arange(0,npos) # should not matter. 
		y[n,0:npos,:] = x[indx, :]
		# add positional encoding
		y[n,0:npos,4] = lin
		y[n,0:npos,5] = 1 # search over these
		curs = np.random.randint(0,npos)
		# print("cursor",curs)
		y[n,npos,4] = curs
		y[n,npos,6] = 1 # cursor token
		y[n,npos+1,7] = 1 # spare token?
		y[n,npos+2,8] = 1 # spare token?
		y[n,npos+3,9] = 1 # reward token / target
		
		# distance output on y[:,-1,4]
		target[n] = abs(curs - torch.argmin(indx)) # we're matching to the zero digit. 
	return y,target
	
	
class ResidualAttentionBlock(nn.Module): 
	def __init__(self, d_model: int, n_head: int, init_zeros:bool):
		super().__init__()
		
		self.n_head = n_head
		self.d_model = d_model
		self.init_zeros = init_zeros
		
		self.wq = LinearM(d_model, n_head*d_model, init_zeros) 
		self.wv = LinearM(d_model, n_head*d_model, init_zeros)
		self.wk = torch.nn.Parameter( 0.005 * torch.ones(n_head, d_model) )
		self.wa = torch.nn.Parameter( 0.5 * torch.ones(n_head) )
		
		self.l1a_f = l1attn_cuda.L1Attn() # dense or full attention
		self.soft = torch.nn.Softmax(dim=2) # unused with L1 attn
		self.fanout = LinearM(d_model, d_model * 1, False)
		self.fanin = LinearM(d_model*2, d_model, False) # this doesn't work?!
		self.gelu = QuickGELU()
		# self.gelu = nn.ReLU()
		# self.gelu = nn.LeakyReLU()

	def fixedInit(self):
		with torch.no_grad():
			n_head = self.n_head
			d_model = self.d_model
			self.wq.w = torch.nn.Parameter(torch.zeros(n_head, d_model, d_model+1))
			self.wk = torch.nn.Parameter(25*torch.ones(n_head, d_model))
			self.wv.w = torch.nn.Parameter(torch.zeros(n_head, d_model, d_model+1))
			# first head: copy the pos enc of the zero token
			self.wq.w[0,5,9] = 25.0

			self.wk[0,4] = 0 # ignore the position

			self.wv.w[0,0,4] = -2.0
			self.wv.w[0,1,4] = 2

			# second head: copy the pos enc of the cursor token
			self.wq.w[1,6,9] = 25.5

			self.wk[1,4] = 0 # ignore the position

			self.wv.w[1,0,4] = 2
			self.wv.w[1,1,4] = -2.1

			self.wq.w = torch.nn.Parameter(self.wq.w.reshape(n_head * d_model, d_model+1))
			self.wv.w = torch.nn.Parameter(self.wv.w.reshape(n_head * d_model, d_model+1))

			# add the two heads post-nonlinearity
			self.fanout.w = torch.nn.Parameter(torch.zeros(d_model, d_model+1))
			self.fanout.w[9,0] = 1 # softmax scales by 0.5
			self.fanout.w[9,1] = 1.0
			self.fanout.w[9,10] = -1 # bias

		
	def attention(self, x:torch.Tensor, axs):
		# pdb.set_trace()
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		width = x.shape[2]
		
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		
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
		
		# extract all global / all-to-all tokens
		# really could do this with pure sparse attn.. will have to compare. 
		a2len = q.shape[1]
		# pad out to BLKSIZ tokens (for CUDA kernel).
		padn = ((a2len + 15) // 16) * 16 - a2len
		assert(padn > 0) # for noop
		qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
		a = a[:, :a2len+1, :a2len, :]
		a[:, a2len, :,:] = 0.0 # slight improvement.. 
		# add in e^0=1 as a 'noop' option
		# (hence max attention is 0.5, not 1)
		# output is b,src,dst,heads
		if False:
			a1 = F.softmax(a, 1) # see l1attn.py -- sm over src
			a2 = F.softmax(a, 2)
			a1 = a1[:, :a2len, :a2len, :]
			a2 = a2[:, :a2len, :a2len, :] # remove noop
			wa = self.wa.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size,ntok,ntok,n_head)
			a = a1 * wa + a2 * (1-wa)
		else:
			a = F.softmax(a, 1) # see l1attn.py -- sm over src FIXME 1
			a = a[:, :a2len, :a2len, :] # remove noop
		b = torch.einsum('bsdh, bshw -> bdhw', a, v)
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		
		if axs is not None: 
			for h in range(self.n_head):
				im = axs[0,h+1].imshow(qq[0,:,h,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[0,h+1])
				axs[0,h+1].set_title(f"qq")
				
				im = axs[1,h+1].imshow(kk[0,:,h,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[1,h+1])
				axs[1,h+1].set_title(f"kk")
				
				im = axs[2,h+1].imshow(v[0,:,h,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[2,h+1])
				axs[2,h+1].set_title(f"v")
				
				im = axs[3,h+1].imshow(a[0,:,:,h].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[3,h+1])
				axs[3,h+1].set_title(f"attn post softmax")
				
				im = axs[4,h+1].imshow(b[0,:,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[4,h+1])
				axs[4,h+1].set_title(f"output b")
			
		return b

	def forward(self, x:torch.Tensor, axs=None):
		y = self.attention(x, axs)
		y = self.gelu(y)
		y = self.fanout(y) # allow sign inversions & mixing; no dim change
		# y = self.gelu(y)
		# y = self.fanin(y)
		return x + y
		
	def plot(self, x): 
		fig,axs = plt.subplots(5, self.n_head+1, figsize=(20,20))
		h = 0
		im = axs[0,h].imshow(self.wq.w.detach().cpu().numpy())
		plt.colorbar(im, ax = axs[0,h])
		axs[0,h].set_title(f"query_{h}")
		
		im = axs[1,h].imshow(self.wk.detach().cpu().numpy())
		plt.colorbar(im, ax = axs[1,h])
		axs[1,h].set_title(f"key_{h}")
	
		im = axs[2,h].imshow(self.wv.w.detach().cpu().numpy())
		plt.colorbar(im, ax = axs[2,h])
		axs[2,h].set_title(f"value_{h}")

		im = axs[3,h].imshow(x[0,:,:].detach().cpu().numpy())
		plt.colorbar(im, ax = axs[3,h])
		axs[3,h].set_title(f"x")
		
		y = self.forward(x, axs)

		im = axs[4,h].imshow(y[0,:,:].detach().cpu().numpy())
		plt.colorbar(im, ax = axs[4,h])
		axs[4,h].set_title(f"y")
		
		plt.show()
	
	
class Transformer(nn.Module): 
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int, init_zeros:bool):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.resblocks = nn.ModuleList([ResidualAttentionBlock(d_model, n_head, init_zeros) for _ in range(layers)])

	def forward(self, x:torch.Tensor):
		for i in range(self.repeat): 
			for j, layer in enumerate(self.resblocks):
				# linearly encode the repeat position on all tokens. 
				# x[:,:,0] = i*2
				x = layer(x)
		return x

	def plot(self, x): 
		for j, layer in enumerate(self.resblocks):
			layer.plot(x)

	def fixedInit(self):
		for layer in self.resblocks:
			layer.fixedInit()
	
if __name__ == '__main__':
	if True:
		fig,axs = plt.subplots(3, 3, figsize=(20,20))
		for i in range(3): 
			for j in range(3):
				y,target = genData(1)
				im = axs[i,j].imshow(y.squeeze().numpy())
				plt.colorbar(im, ax = axs[i,j])
				axs[i,j].set_title(f"target:{target.item()}")
		plt.show()
	
	model = Transformer(d_model=width, layers=1, repeat=1, n_head=2, init_zeros=False)
	# model.fixedInit()
	model = model.cuda()

	use_adam = True
	
	if use_adam:
		optimizer = optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
	else: 
		optimizer = psgd.LRA(model.parameters(),lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,\
			preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=20, grad_clip_max_norm=5.0)
	
	fd_losslog = open('losslog.txt', 'w')
	
	for i in range(500000):
		x,target = genData(batch_size)
		x = x.cuda()
		target = target.cuda()
		if use_adam:
			y = model(x)
			loss = torch.sum( (y[:,-1,-1] - target)**2 )
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
			loss.backward()
			optimizer.step()
		else: 
			def closure(): 
				y = model(x)
				loss = torch.sum( (y[:,-1,-1] - target)**2 ) + \
						sum( \
						[torch.sum(1e-4 * torch.rand_like(param) * param * param ) for param in model.parameters()])
				return loss
			loss = optimizer.step(closure) 
		lloss = loss.detach().cpu().item()
		print(lloss)
	
		fd_losslog.write(f'{i}\t{lloss}\n')
		fd_losslog.flush()

	x,target = genData(batch_size)
	x = x.cuda()
	y = model.plot(x)
