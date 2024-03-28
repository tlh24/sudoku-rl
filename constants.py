n_heads = (1*4)+1
world_dim = 20 # 36, must be even!
xfrmr_dim = 20 # default: 128
reward_dim = 1 # immediate and infinite-horizon
board_cnt = 5
token_cnt = board_cnt + 3 # 5+3; used to be:152
g_zeroinit = True
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
