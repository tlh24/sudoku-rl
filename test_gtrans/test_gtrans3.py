# want to test the graph transformer on a few basic tasks
# e.g. is an element in the set, 
# what is the distance to match a given pattern.  
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import l1attn_cuda
import matplotlib.pyplot as plt
import pdb
import psgd 
import time

# this is a 2d version of test_gtrans2

npos = 3
npos2 = npos*npos
ntok = npos2 + 4
width = 16
sample_size = 128 # sample_size target values for 782 unknowns! 

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)
	  
class LinearM(nn.Module): 
	# with the bias merged -- used for PSGD optimizer.
	def __init__(self, indim:int, outdim:int, initzeros:bool): 
		super(LinearM, self).__init__()
		scl = 0.005
		if initzeros: 
			self.w = torch.nn.Parameter( scl * torch.ones(outdim,indim+1))
		else:
			self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim+1))
		with torch.no_grad(): 
			self.w[:,-1] = 0.0 # bias starts at 0
		
	def forward(self, x):
		return torch.einsum('oi,bhi -> bho', self.w[:,:-1], x) + self.w[:,-1]

def genData(nn): 
	y = torch.zeros(nn, ntok, width)
	target = torch.zeros(nn,2)
	lin = torch.arange(0,npos2)
	x = torch.zeros(npos2, width)
	posx = torch.arange(0,npos).unsqueeze(1).tile(1,npos)
	posy = torch.arange(0,npos).unsqueeze(0).tile(npos,1)
	pos = torch.zeros(npos, npos, 2)
	pos[:,:,0] = posx
	pos[:,:,1] = posy
	pos = torch.reshape(pos, (npos2,2))
	# binary encoding of the digits.  
	x[:,0] = (lin % 2) * 10
	x[:,1] = ((lin//2) % 2) * 10
	x[:,2] = ((lin//4) % 2) * 10
	x[:,3] = ((lin//8) % 2) * 10
	x[:,4] = ((lin//16) % 2) * 10
	for n in range(nn): 
		# shuffle tokens
		indx = torch.randperm(npos2)
		# print(indx)
		# indx = torch.arange(0,npos) # should not matter. 
		y[n,0:npos2,:] = x[indx, :]
		# add positional encoding
		y[n,0:npos2,5:7] = pos
		y[n,0:npos2,7] = 10 # search over these
		curs = np.random.randint(0,npos, (2,))
		# print("cursor",curs)
		y[n,npos2,5] = curs[0]
		y[n,npos2,6] = curs[1]
		y[n,npos2,8] = 10 # cursor token
		y[n,npos2+1,9] = 10 # spare token?
		y[n,npos2+2,10] = 10 # spare token?
		y[n,npos2+3,11] = 10 # reward token / target
		
		# distance output on y[:,-1,4]
		zero_indx = torch.argmin(indx)
		target[n,0] = torch.abs(pos[zero_indx,0] - curs[0])
		target[n,1] = torch.abs(pos[zero_indx,1] - curs[1])
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
		# self.wa = torch.nn.Parameter( 1.0 * torch.ones(n_head) )
		
		self.l1a_f = l1attn_cuda.L1Attn() # dense or full attention
		self.soft = torch.nn.Softmax(dim=2) # unused with L1 attn
		self.fanout = LinearM(d_model, d_model * 1, False)
		# self.fanin = LinearM(d_model*2, d_model, False) # this doesn't work?!
		# self.gelu = QuickGELU()
		self.gelu = nn.ReLU()
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
			self.wv.w[1,1,4] = -2.0

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
		
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		width = x.shape[2]
		
		q = self.wq(x)
		q = torch.reshape(q, (batch_size, ntok, self.n_head, d_head))
		# wa = self.wa.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
		# q = q # * wa
		v = self.wv(x)
		v = torch.reshape(v, (batch_size, ntok, self.n_head, d_head))
		
		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		
		gk = self.wk.unsqueeze(0).unsqueeze(0)
		# bk = self.bk.w.unsqueeze(0).unsqueeze(0).expand([batch_size,ntok,-1,-1]) # bias term, not needed.
		k = k * gk # * wa
		
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
			wa = self.wa.unsqueeze(0).unsqueeze(0).unsqueeze(0)
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
			
	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params}")
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', type=int, default=128, help='sample size')
	parser.add_argument('-d', type=int, default=0, help='CUDA device')
	parser.add_argument('--layers', type=int, default=1, help='number of layers')
	parser.add_argument('--heads', type=int, default=1, help='number of heads')
	parser.add_argument('-a', action='store_true', help='use AdamW')
	parser.add_argument('-m', action='store_true', help='many-mode')
	cmd_args = parser.parse_args()
	sample_size = cmd_args.b
	
	if True:
		fig,axs = plt.subplots(3, 3, figsize=(20,20))
		for i in range(3): 
			for j in range(3):
				y,target = genData(1)
				im = axs[i,j].imshow(y.squeeze().numpy())
				plt.colorbar(im, ax = axs[i,j])
				axs[i,j].set_title(f"target:{target[0,0].item()},{target[0,1].item()}")
		plt.show()
	
	
	start_time = time.time()
	duration = 600
	j = 0
	while j < 200: 
		
		model = Transformer(d_model=width, layers=cmd_args.layers, repeat=1, n_head=cmd_args.heads, init_zeros=False)
		model.printParamCount()
		# pdb.set_trace()
		# model.fixedInit()
		model = model.cuda(cmd_args.d)

		use_adam = False # cmd_args.a
		
		if use_adam:
			optimizer = optim.AdamW(model.parameters(), lr=2e-3, amsgrad=True)
		else: 
			optimizer = psgd.LRA(model.parameters(),lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,\
				preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=10, grad_clip_max_norm=5.0)
		
		fd_losslog = open('losslog.txt', 'w')
		
		dat,_ = genData(2000)
		mean = torch.sum(dat, (0,1)) / (2000 * ntok)
		std = torch.sqrt(torch.sum((dat - mean)**2, (0,1)) / (2000 * ntok))
		std = std / 3.5 # help l1 attn select one
		
		x,target = genData(sample_size)
		# x = (x - mean) / std # learn the affine transform later
		x = x.cuda(cmd_args.d)
		target = target.cuda(cmd_args.d)
		slowloss = 1e6
		slowdeltaloss = 1
		
		for i in range(150000): # fixme: 15000
			if use_adam:
				y = model(x)
				loss = torch.sum( (y[:,-1,14:16] - target)**2 )
				torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
				loss.backward()
				optimizer.step()
			else: 
				def closure(): 
					y = model(x)
					loss = torch.sum( (y[:,-1,14] - target[:,0])**2 ) + \
							sum( \
							[torch.sum(2.5e-4 * torch.rand_like(param) * torch.abs(param) ) for param in model.parameters()])
					return loss
				loss = optimizer.step(closure) 
			lloss = loss.detach().cpu().item()
			slowloss2 = 0.94 * slowloss + 0.06 * (lloss / 32)
			slowdeltaloss = 0.99 * slowdeltaloss + 0.01 * abs(slowloss - slowloss2)
			slowloss = slowloss2
			if not cmd_args.m: 
				if i % 10 == 0: 
					print(lloss, slowloss, slowdeltaloss)
					fd_losslog.write(f'{i}\t{lloss}\t{slowloss}\n')
					fd_losslog.flush()
			# if slowdeltaloss < 2e-6:
			# 	break # conservatively converged, start a new model.
			elapsed_time = time.time() - start_time
			if elapsed_time > duration :
				j = 200
				break
		j = j + 1
			
	
		x,target = genData(1000)
		# x = (x - mean) / std # learn the affine transform later
		x = x.cuda(cmd_args.d)
		target = target.cuda(cmd_args.d)
		y = model(x)
		loss = torch.sum( (y[:,-1,14:16] - target)**2 )
		lloss = loss.detach().cpu().item()
		print("v",lloss)
		fd_losslog.write(f'{i}\t{lloss}\t{0.0}\n')
		fd_losslog.flush()
	
		if cmd_args.m: 
			fd_vallog = open(f'vallog3_l{cmd_args.layers}_h{cmd_args.heads}.txt', 'a')
			fd_vallog.write(f'{sample_size}\t{lloss/1000}\n') 
			fd_vallog.flush()
			fd_vallog.close()
	
	if not cmd_args.m: 
		x,target = genData(sample_size)
		x = x.cuda(cmd_args.d)
		y = model.plot(x)
