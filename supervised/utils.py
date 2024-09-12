import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import logging


def isValidSudoku(board) -> bool:
    if isinstance(board, np.ndarray):
        board = board.tolist()
    
    for i in range(9):
        row = board[i]
        if len(row)!=len(set(row)): return False
        col = [board[c][i] for c in range(9)]
        if len(col)!=len(set(col)): return False
        box = [board[ind//3+(i//3)*3][ind%3+(i%3)*3] for ind in range(9)]
        if len(box)!=len(set(box)): return False
    return True
                

def configure_optimizer(model, weight_decay, learning_rate, betas):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add('pos_emb')

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


class Sudoku_Dataset_SATNet(Dataset):
	def __init__(self):
		data = {}
		data_to_path = {
			'board': './satnet/features.pt',
			'board_img': './satnet/features_img.pt',
			'label': './satnet/labels.pt',
			'perm': './satnet/perm.pt',
		}
		for k in data_to_path:
			try:
				with open(data_to_path[k], 'rb') as f:
					data[k] = torch.load(f)
			except Exception as error:
				print(f"could not find data file: {error}")
				print("please download: wget -cq powei.tw/sudoku.zip && unzip -qq sudoku.zip (see https://github.com/azreasoners/recurrent_transformer)")
		# board has shape (10000, 81), 0's with no digits and 1-9 for digits
		self.board = ((data['board'].sum(-1) != 0) * (data['board'].argmax(-1) + 1)).view(-1, 81).long()
		self.label = data['label'].argmax(-1).view(-1, 81).long() # (10000, 81)
		self.label_ug = self.label.clone() # (10000, 81)
		# label_ug is a label vector of indices(0-8 vs 1-9) of size 81 but all initially given digits are -100
		self.label_ug[self.board != 0] = -100

	def __len__(self):
		return len(self.board)

	def __getitem__(self, idx):
		"""
		Each data instance is a tuple <board, board_img, label, label_ug> where
			board: a float tensor of shape (81) consisting of {0,...,9}
			label_ug: a float tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
		"""
		# return self.board[idx], self.board_img[idx], self.label[idx], self.label_ug[idx]
		return self.board[idx], self.label_ug[idx]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
