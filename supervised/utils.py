import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import math 


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class TrainerConfig:
    num_epochs = 100
    batch_size = 64
    learning_rate = 3e-4
    grad_norm_clip = 1.0
    val_interval = 10
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                logits, loss = self.model(x, y) 
                loss = loss.mean()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                total_train_loss += loss.item()

            # Print epoch results
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_train_loss:.4f}")

            # Validation loss
            if epoch % self.config.val_interval == 0:
                total_val_loss = 0
                self.model.eval()
                batch_accs = []
            
                with torch.no_grad():
                    for batch in self.val_dataloader:
                        x,y = batch 
                        x,y = x.to(self.device), y.to(self.device)

                        logits, loss = self.model(x, y)
                        loss = loss.mean()
                        total_val_loss += loss.item() 
                        batch_accs.append(Trainer.return_accuracy(logits, y))

                avg_val_loss = total_val_loss / len(self.val_dataloader)
                avg_avg_acc = np.mean(batch_accs)
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}, Val Loss: {avg_val_loss:.4f}\
                        Val Avg Avg Acc: {avg_avg_acc:.4f}")

                # Save the best model 
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print(f'New best model saved with val loss {best_val_loss}')
        print("Training completed.")

    def evaluate(self):
        breakpoint()
        self.model.eval()
        total_eval_loss = 0
        batch_accs = []

        with torch.no_grad():
            for batch in self.test_dataloader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                logits, loss = self.model(x, y)
                loss = loss.mean()
                total_eval_loss += loss.item()
                batch_accs.append(Trainer.return_accuracy(logits, y))

        avg_eval_loss = total_eval_loss / len(self.test_dataloader)
        avg_avg_accuracy = np.mean(batch_accs) 
        print(f"Avg_eval_loss is {avg_eval_loss:.4f} Avg Avg Acc: {avg_avg_accuracy:.4f}")
        return avg_avg_accuracy
    
    @staticmethod
    def return_accuracy(batch_logits, batch_targets):
        total_predicted = 0
        total_correct = 0
        for batch_idx in range(len(batch_logits)):
            logits = batch_logits[batch_idx]
            targets = batch_targets[batch_idx]
            for i in range(0, len(targets)):
                # only count accuracy on initially empty cells
                if targets[i] > 0:
                    predicted_num = torch.argmax(logits[i]) 
                    total_predicted += 1
                    total_correct += int(predicted_num == targets[i])
        
        return total_correct/total_predicted


                
            
            


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.C = self.f = self.create_v = self.tok_emb = None
        for k,v in kwargs.items():
            setattr(self, k, v)

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    We are doing full-attention, no causal self attention here
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        if isinstance(x, tuple):
            x = x[0]
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #if self.causal_mask:
        #    att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att_to_check = att.clone()
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att_to_check


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        att, att_to_check = self.attn(self.ln1(x))
        x = x + att 
        x = x + self.mlp(self.ln2(x))
        return x, att_to_check


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
        loss = None
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
			try:
				with open(data_to_path[k], 'rb') as f:
					data[k] = torch.load(f)
			except Exception as error:
				print(f"could not find data file: {error}")
				print("please download from https://github.com/azreasoners/recurrent_transformer")
				exit()
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
