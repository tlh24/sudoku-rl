xfrmr_width = 128
world_dim = 1 + 9*3 + 6 # must be even!
action_dim = 10 + 9 
	# digits 0-9 (0=nothing); move, set/unset, note/unnote, nop
reward_dim = 2 # immediate and infinte-horizon
latent_cnt = 96 - 81 - 1 # 14

batch_size = 16
