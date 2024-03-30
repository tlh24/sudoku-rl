n_heads = (2*4)+1
world_dim = 20 # 36, must be even!
xfrmr_dim = 20 # default: 128
reward_dim = 1 # immediate and infinite-horizon
token_cnt = 8 # run graph_encoding to determine this. 
g_zeroinit = False
g_l1atten = True
g_globalatten = False

batch_size = 128

# sudoku board size
if True: 
	SuN = 4 # 4 x 4 board: 4 entries per row, column and box.
	SuH = 2
	SuK = 5
else: 
	SuN = 9 # 4 x 4 board: 4 entries per row, column and box.
	SuH = 3 # sqrt SuN
	SuK = 25
