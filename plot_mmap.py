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

if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description='image mmaped files')
	# parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int)
 # 
	# args = parser.parse_args()
	# batch_size = args.batch_size
 # 
	print(f"batch_size:{batch_size}")

	fd_board = make_mmf("board.mmap", [batch_size, 82, world_dim])
	fd_new_board = make_mmf("new_board.mmap", [batch_size, 82, world_dim])
	fd_worldp = make_mmf("worldp.mmap", [batch_size, 82, world_dim])
	fd_action = make_mmf("action.mmap", [batch_size, latent_cnt, action_dim])
	fd_actionp = make_mmf("actionp.mmap", [batch_size, latent_cnt, action_dim])
	fd_reward = make_mmf("reward.mmap", [batch_size, latent_cnt, reward_dim])
	fd_rewardp = make_mmf("rewardp.mmap", [batch_size, latent_cnt, reward_dim])
	fd_latent = make_mmf("latent.mmap", [batch_size, latent_cnt, world_dim])


	plot_rows = 2
	plot_cols = 5
	figsize = (16, 8)
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	initialized = False
	im = [ [0]*plot_cols for i in range(plot_rows)]
	cbar = [ [0]*plot_cols for i in range(plot_rows)]


	def plot_tensor(r, c, v, name, lo, hi):
		if not initialized:
			# seed with random data so we get the range right
			cmap_name = 'PuRd' # purple-red
			if lo == -1*hi:
				cmap_name = 'seismic'
			data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
			data = np.reshape(data, (v.shape[0], v.shape[1]))
			im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
			cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
		im[r][c].set_data(v.numpy())
		#cbar[r][c].update_normal(im[r][c]) # probably does nothing
		axs[r,c].set_title(name)
		axs[r,c].tick_params(bottom=True, top=True, left=True, right=True)

	bs = batch_size
	if batch_size > 32: 
		bs = 32
		
	update = True

	while True:
		if update: 
			board = read_mmap(fd_board, [batch_size, 82, world_dim])
			new_board = read_mmap(fd_new_board, [batch_size, 82, world_dim])
			worldp = read_mmap(fd_worldp, [batch_size, 82, world_dim])
			action = read_mmap(fd_action, [batch_size, latent_cnt, action_dim])
			actionp = read_mmap(fd_actionp, [batch_size, latent_cnt, action_dim])
			reward = read_mmap(fd_reward, [batch_size, latent_cnt, reward_dim])
			rewardp = read_mmap(fd_rewardp, [batch_size, latent_cnt, reward_dim])
			latent = read_mmap(fd_latent, [batch_size, latent_cnt, world_dim])

		# i = np.random.randint(batch_size) # checking
		i = 0
		plot_tensor(0, 0, new_board[i,:,:], f"new_board[{i},:,:]", -2.0, 2.0)
		plot_tensor(1, 0, worldp[i,:,:], f"worldp[{i},:,:]", -2.0, 2.0)
		plot_tensor(0, 1, new_board[i,:,:] - board[i,:,:], f"(new_board -  board)[{i},:,:]", -2.0, 2.0)
		plot_tensor(1, 1, worldp[i,:,:] - board[i,:,:], f"(worldp - board)[{i},:,:]", -2.0, 2.0)
		plot_tensor(0, 2, action[i,:,:], f"action[{i},:,:]", -2.0, 2.0)
		plot_tensor(1, 2, actionp[i,:,:], f"actionp[{i},:,:]", -2.0, 2.0)
		plot_tensor(0, 3, np.transpose(latent[i,:,:]), f"latent (slow and fast)", -2.0, 2.0)
		plot_tensor(1, 3, actionp[i,:,:] - action[i,:,:], f"(actionp - action)[{i},:,:]", -2.0, 2.0)
		plot_tensor(0, 4, reward[i,:,:], f"reward[{i},:,:]", -2.0, 2.0)
		plot_tensor(1, 4, rewardp[i,:,:], f"rewardp[{i},:,:]", -2.0, 2.0)
		
		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()
		# time.sleep(2)
		print("tock")
		initialized=True
		
		# if reward[i,0,-1] - rewardp[i,0,-1] > 0.9: 
		# 	print('paused!')
		# 	update = False
			

