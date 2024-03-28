import math
import mmap
import numpy as np
import torch as th
import argparse
import matplotlib.pyplot as plt
from ctypes import * # for c_char
import time
import os
import io
import pdb
from constants import *

# remove menubar buttons
plt.rcParams['toolbar'] = 'None'

def make_mmf(fname, dims): 
	if not os.path.isfile(fname):
		siz = math.floor(np.prod(dims)) * 4 
		os.system(f"fallocate -l {siz} {fname}")
	fd = open(fname, "r+b")
	return mmap.mmap(fd.fileno(), 0)

def read_mmap(mmf, dims): 
	mmf.seek(0)
	mmb = mmf.read()
	# siz = len(mmb)
	siz = math.prod(dims) * 4
	mmb2 = (c_char * siz).from_buffer_copy(mmb)
	x = th.frombuffer(mmb2, dtype=th.float).clone()
	x = th.reshape(x, dims)
	return x
	
def write_mmap(mmf, data): 
	buff = io.BytesIO()
	data = data.to(th.float32)
	buff = data.numpy().tobytes()
	mmf.seek(0)
	n = mmf.write(buff)
	return n
	
def zscore(x): 
	s = th.std(x)
	m = th.mean(x)
	return x / s
	
def zscale(x): 
	return x / th.max(x)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", type=int, choices=range(0,2), default=0, help="set the display mode. 0 = model output, 1 = wqv")
	args = parser.parse_args()
	mode = args.mode
	
	print(f"batch_size:{batch_size}")

	fd_board = make_mmf("board.mmap", [batch_size, board_cnt, world_dim])
	fd_new_board = make_mmf("new_board.mmap", [batch_size, board_cnt, world_dim])
	fd_boardp = make_mmf("boardp.mmap", [batch_size, board_cnt, world_dim])
	fd_reward = make_mmf("reward.mmap", [batch_size, reward_dim])
	fd_rewardp = make_mmf("rewardp.mmap", [batch_size, reward_dim])
	fd_attention = make_mmf("attention.mmap", [2, token_cnt, token_cnt, n_heads])
	fd_wqkv = make_mmf("wqkv.mmap", [n_heads*2,2*xfrmr_dim,xfrmr_dim])

	if mode == 0: 
		plot_rows = 2
		plot_cols = 3
		figsize = (12, 6)
	if mode == 1: 
		plot_rows = 4
		plot_cols = 9
		figsize = (32, 12)
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	initialized = False
	im = [ [0]*plot_cols for i in range(plot_rows)]
	cbar = [ [0]*plot_cols for i in range(plot_rows)]


	def plot_tensor(r, c, v, name, lo, hi, colorbar=True):
		if not initialized:
			# seed with random data so we get the range right
			cmap_name = 'PuRd' # purple-red
			if lo == -1*hi:
				cmap_name = 'seismic'
			data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
			data = np.reshape(data, (v.shape[0], v.shape[1]))
			im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
			if colorbar: 
				cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
		im[r][c].set_data(v.numpy())
		#cbar[r][c].update_normal(im[r][c]) # probably does nothing
		axs[r,c].set_title(name)
		axs[r,c].tick_params(bottom=True, top=True, left=True, right=True)

	bs = batch_size
	if batch_size > 4: 
		bs = 4
		
	u = 0
	
	maxattn = th.ones(n_heads*2)
	maxqkv = th.ones(n_heads*2)

	while True:
		# i = np.random.randint(batch_size) # checking
		i = 0
		cl = 40
			
		if mode == 0: 
			board = read_mmap(fd_board, [batch_size, board_cnt, world_dim])
			new_board = read_mmap(fd_new_board, [batch_size, board_cnt, world_dim])
			boardp = read_mmap(fd_boardp, [batch_size, board_cnt, world_dim])
			reward = read_mmap(fd_reward, [batch_size, reward_dim])
			rewardp = read_mmap(fd_rewardp, [batch_size, reward_dim])
			
			plot_tensor(0, 0, new_board[i,:cl,:].T, f"new_board[{i},:,:]", -4.0, 4.0)
			plot_tensor(1, 0, boardp[i,:cl,:].T, f"worldp[{i},:,:]", -4.0, 4.0)
			plot_tensor(0, 1, new_board[i,:cl,:].T - board[i,:cl,:].T, f"(new_board -  board)[{i},:,:]", -4.0, 4.0)
			plot_tensor(1, 1, boardp[i,:cl,:].T - board[i,:cl,:].T, f"(worldp - board)[{i},:,:]", -4.0, 4.0)
			plot_tensor(0, 2, reward[:,:], f"reward[{i},:,:]", -2.0, 2.0)
			plot_tensor(1, 2, rewardp[:,:], f"rewardp[{i},:,:]", -2.0, 2.0)
			
		if mode == 1: 
			
			attention = read_mmap(fd_attention, [2, token_cnt, token_cnt, n_heads])
			wqkv = read_mmap(fd_wqkv, [2,n_heads,2*xfrmr_dim,xfrmr_dim])

			for layer in range(2): 
				for head in range(n_heads): 
					x = attention[layer,:cl,:cl,head]
					# j = layer*n_heads + head
					# maxattn[j] = maxattn[j] * 0.97 + th.max(x) * 0.03
					# x = x / maxattn[j]
					plot_tensor(layer+0, head, x, f"attention[{layer},:,:,{head}]", -1.0, 1.0, colorbar=False)
					
				for head in range(n_heads): 
					x = wqkv[layer,head,:,:]
					maxqkv[head] = maxqkv[head] * 0.97 + th.max(th.abs(x)) * 0.03
					maxqkv[head] = th.clamp(th.abs(maxqkv[head]), 0.01, 1e6)
					x = x.T / maxqkv[head]
					
					plot_tensor(layer+2, head, x, f"wqv[{layer},{head},:,:]", -1.0, 1.0, colorbar=False)
					if not initialized: 
						axs[layer+2,head].plot([19.5, 19.5], [0.0, 19.5], 'g')
					if head == 0: 
						axs[layer+2,head].set_ylabel('input dim')
						axs[layer+2,head].set_xlabel('output dim for Q & V')
				
		
		
		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()
		# time.sleep(0.35)
		print("tock")
		initialized=True
		u = u + 1
		
		# if reward[i,0,-1] - rewardp[i,0,-1] > 0.9: 
		# 	print('paused!')
		# 	update = False
			

