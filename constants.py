n_heads = 3
world_dim = 1 + 9*3 + 8 # 36, must be even!
# 1 indicates presence of cursor
# 9 are the digits 1..9; there are 3 of these:
# supplied number
# agent guess
# agent notes
# 8 positional encodes:
# -- Sin/cos for x,y,block
xfrmr_dim = 64 # default: 128
action_dim = 10 + 9 
latent_dim = xfrmr_dim - world_dim - action_dim
	# digits 0-9 (0=nothing); move, set/unset, note/unnote, nop
reward_dim = 2 # immediate and infinite-horizon
token_cnt = 96
latent_cnt = token_cnt - 82 # 14
batch_size = 32
sudoku_width = 9
