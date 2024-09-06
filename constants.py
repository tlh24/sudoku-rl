import torch

n_heads = 6
world_dim = 64 #
xfrmr_dim = 64 # default: 128
reward_dim = 1 # immediate and infinite-horizon
token_cnt = 114 # 114 # run graph_encoding to determine this.
g_zeroinit = False
g_l1atten = True
g_globalatten = False
g_dtype = torch.float32 # March 6: I can't get float16 to work stably.
duration = 256 # length of rollouts

batch_size = 64 # 128 does not seem to go any faster?

# sudoku board size
if False:
	SuN = 4 # 4 x 4 board: 4 entries per row, column and box.
	SuH = 2
	SuK = 5
else: 
	SuN = 9 # 4 x 4 board: 4 entries per row, column and box.
	SuH = 3 # sqrt SuN
	SuK = 25

### Utility functions

def plot_tensor(v, name, lo, hi):
	''' this is used primarily for debugging '''
	cmap_name = 'seismic'
	fig, axs = plt.subplots(1, 1, figsize=(12,6))
	data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
	data = np.reshape(data, (v.shape[0], v.shape[1]))
	im = axs.imshow(data, cmap = cmap_name)
	plt.colorbar(im, ax=axs)
	im.set_data(v)
	axs.set_title(name)
	axs.tick_params(bottom=True, top=True, left=True, right=True)
	plt.show()
