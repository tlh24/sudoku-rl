import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head 
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_classes, bias=False)
        ## TODO: check that I can skip a bunch of other things
        
        self.block_size = config.block_size
        self.n_recur = config.n_recur
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        '''
        Returns:
            the loss as a scalar
            the logits in the final prediction of shape (batch_size, 81, 9)
        '''
        
        b, t = idx.shape
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each idx maps to a learned vector
        position_embeddings = self.pos_emb[:, :t, :] # each pos maps to a learned vector
        x = self.drop(token_embeddings + position_embeddings)
        # collect the attention matrices and last layer predicted logits
        atts = []
        for _ in range(self.n_recur):
            for block in self.blocks:
                x, att = block(x) # (batch, 81, 128) (batch, num_heads, 81, 81)
                atts.append(att)

        logits = self.head(self.ln_f(x))
        # compute losses
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # TODO: Add Sudoku-specific constraint loss or attention loss here if needed

        return logits, loss 
        #return logits, loss, torch.stack(atts)

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
            with open(data_to_path[k], 'rb') as f:
                data[k] = torch.load(f)
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