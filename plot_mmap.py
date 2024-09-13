import math
import mmap
import numpy as np
import torch as th
import argparse
import matplotlib
import matplotlib.pyplot as plt
from ctypes import c_char
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
	fsiz = len(mmb)
	siz = math.prod(dims) * 4
	if fsiz < siz:
		print('memory-mapped file too small.  rm *.mmap files and try again.')
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

	fd_board = make_mmf("board.mmap", [batch_size, token_cnt, world_dim])
	fd_new_board = make_mmf("new_board.mmap", [batch_size, token_cnt, world_dim])
	fd_boardp = make_mmf("boardp.mmap", [batch_size, token_cnt, world_dim])
	fd_reward = make_mmf("reward.mmap", [batch_size, reward_dim])
	fd_rewardp = make_mmf("rewardp.mmap", [batch_size, reward_dim])
	fd_attention = make_mmf("attention.mmap", [2, token_cnt, token_cnt, n_heads])
	fd_wqkv = make_mmf("wqkv.mmap", [n_heads*2,2*xfrmr_dim,xfrmr_dim])

	if "DISPLAY" not in os.environ:
		print("No X11 server detected, switching to non-interactive Agg backend")
		matplotlib.use('Agg')  # Use non-interactive backend

	if mode == 0: 
		plot_rows = 2
		plot_cols = 3
		# figsize = (12, 6)
	if mode == 1: 
		plot_rows = 4
		plot_cols = 9
	figsize = (32, 12)
	plt.ion()
	plt.rcParams['toolbar'] = 'toolbar2'
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	initialized = False
	im = [ [0]*plot_cols for i in range(plot_rows)]
	cbar = [ [0]*plot_cols for i in range(plot_rows)]
	
	current_directory = os.getcwd()
	base_dir = os.path.basename(current_directory)
	fig.canvas.manager.set_window_title(f'plot_mmap {base_dir}')

	def plotTensor(r, c, v, name, lo, hi, colorbar=True):
		if not initialized:
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
		
	u = 0
	cl = token_cnt
	# cl = 120
	
	maxattn = th.ones(n_heads*2)
	maxqkv = th.ones(n_heads*2)
	mask = torch.zeros(token_cnt, world_dim)
	mask[:,:32] = 1.0; 
	lines = None
	cont = True

	while cont:
		# i = np.random.randint(batch_size) # checking
		i = 0
			
		if mode == 0: 
			board = read_mmap(fd_board, [batch_size, token_cnt, world_dim])
			new_board = read_mmap(fd_new_board, [batch_size, token_cnt, world_dim])
			boardp = read_mmap(fd_boardp, [batch_size, token_cnt, world_dim])
			reward = read_mmap(fd_reward, [batch_size, reward_dim])
			rewardp = read_mmap(fd_rewardp, [batch_size, reward_dim])
			
			if u % 2 == 0 or True: 
				err = torch.sum((boardp[:,:,33:] - new_board[:,:,1:32])**2, [1,2])
				i = torch.argmax(err).item()
				maxerr = err[i]
				sumerr = torch.sum(err)
			else: 
				i = 0
			
			guess = 10 + torch.argmax(new_board[i,0,10:20])
			
			plotTensor(0, 0, new_board[i,:cl,:32].T, f"new_board[{i},:,:32]", -4.0, 4.0)
			if lines is not None: 
				lines.pop(0).remove()
			lines = axs[0,0].plot([0,4,0,0,cl-1],[6,6,6,guess,guess], 'g', alpha=0.4)
			plotTensor(1, 0, boardp[i,:cl,:].T, f"board_pred[{i},:,:]", -4.0, 4.0)
			plotTensor(0, 1, new_board[i,:cl,:32].T - board[i,:cl,:32].T, f"(new_board -  board)[{i},:,:32]", -4.0, 4.0)
			plotTensor(1, 1, boardp[i,:cl,:].T - board[i,:cl,:].T, f"(board_pred - board)[{i},:,:]", -4.0, 4.0)
			plotTensor(0, 2, (boardp[i,:cl,33:].T - new_board[i,:cl,1:32].T), f"(board_pred - new_board)[{i},:,:32]", -4.0, 4.0)
			plotTensor(1, 2, torch.cat((reward,rewardp),1), f"reward & rewardp", -5.0, 5.0)
			
		if mode == 1: 
			attention = read_mmap(fd_attention, [2, token_cnt, token_cnt, n_heads])
			wqkv = read_mmap(fd_wqkv, [2,n_heads,2*xfrmr_dim,xfrmr_dim])

			for layer in range(2): 
				for head in range(n_heads): 
					x = attention[layer,:cl,:cl,head]
					# j = layer*n_heads + head
					# maxattn[j] = maxattn[j] * 0.97 + th.max(x) * 0.03
					# x = x / maxattn[j]
					plotTensor(layer+0, head, x, f"attention[{layer},:,:,{head}]", -1.0, 1.0, colorbar=False)
					
				for head in range(n_heads): 
					x = wqkv[layer,head,:,:]
					maxqkv[head] = maxqkv[head] * 0.97 + th.max(th.abs(x)) * 0.03
					maxqkv[head] = th.clamp(th.abs(maxqkv[head]), 0.01, 1e6)
					x = x.T / maxqkv[head]
					
					plotTensor(layer+2, head, x, f"wqv[{layer},{head},:,:]", -1.0, 1.0, colorbar=False)
					# if not initialized: 
					# 	axs[layer+2,head].plot([19.5, 19.5], [0.0, 19.5], 'g')
					if head == 0: 
						axs[layer+2,head].set_ylabel('input dim')
						axs[layer+2,head].set_xlabel('output dim for Q & V')
				
		
		
		fig.tight_layout()
		if matplotlib.get_backend() == 'Agg':
			# Save plot to PNG file when in non-interactive mode
			output_file = 'plot_mmap.png'
			fig.savefig(output_file)
			print(f"Plot saved to {output_file}")
			cont = False
		else:
			fig.canvas.draw()
			fig.canvas.flush_events()
			print(f"tock {maxerr} {sumerr}")
			if maxerr > 1.0 and False:
				time.sleep(60.0)
			else:
				time.sleep(4.0)
		initialized=True
		u = u + 1

