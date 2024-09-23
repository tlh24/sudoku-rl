'''
Code derived from AZReasoners Recurrent Transformer for CSP, which is derived from nano-GPT
'''
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math 
from supervised.utils import save_checkpoint 
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
	def __init__(self, state, train_dataset, val_dataset, test_dataset, config, optimizer, logger):
		self.state = state
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset
		self.config = config
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.logger = logger
		self.model = self.state['model'].to(self.device)
		self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
		self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
		self.test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
		self.optimizer = optimizer

	def train(self):
		best_loss = float('inf')

		for epoch in range(self.config.num_epochs):
			self.model.train()
			total_train_loss = 0

			for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				# Forward pass
				loss = self.model(x, y)

				# Backward pass and optimization
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
				self.optimizer.step()

				total_train_loss += loss.item()

			# Print epoch results
			avg_train_loss = total_train_loss / len(self.train_dataloader)
			print(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {avg_train_loss:.4f}")

			self.state['step'] += 1

			# Validation loss
			if epoch > 0 and epoch % self.config.val_interval == 0:
				total_val_loss = 0
				self.model.eval()

				with torch.no_grad():
					for batch in self.val_dataloader:
						x,y = batch
						x,y = x.to(self.device), y.to(self.device)

						loss = self.model(x, y)
						total_val_loss += loss.item()

				avg_val_loss = total_val_loss / len(self.val_dataloader)
				print(f"Epoch {epoch + 1}/{self.config.num_epochs}, Val Loss: {avg_val_loss:.4f}")
				self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} Val Loss: {avg_val_loss:.4f}")

				# Save the best model
				if avg_val_loss < best_loss:
					best_loss = avg_val_loss
					save_checkpoint(f'epoch{epoch}_best.pth', self.state)
					print(f'New best model saved with loss {best_loss}')
		print("Training completed.")

	def evaluate(self):
		self.model.eval()
		total_eval_loss = 0

		with torch.no_grad():
			for batch in self.test_dataloader:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				loss = self.model(x, y)
				loss = loss.mean()
				total_eval_loss += loss.item()

		avg_eval_loss = total_eval_loss / len(self.test_dataloader)
		print(f"Avg_eval_loss is {avg_eval_loss:.4f}")
		self.logger.info(f"Avg_eval_loss is {avg_eval_loss:.4f}")
		return avg_eval_loss

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
		y, att_to_check = self.attn(self.ln1(x))
		x = x + y
		x = x + self.mlp(self.ln2(x))
		return x, att_to_check


class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		# input embedding
		self.one_hot_emb = nn.Linear(config.vocab_size, config.n_embd)
		self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
		self.drop = nn.Dropout(config.embd_pdrop)
		# transformer
		self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
		# decoder head
		self.ln_f = nn.LayerNorm(config.n_embd)
		self.head = nn.Sequential(nn.Linear(81* config.n_embd, 256, bias=True),
								nn.ReLU(),
							    nn.Linear(256,1),
								nn.Sigmoid()) 
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

	def forward(self, one_hots: torch.Tensor, targets=None):
		'''
		one_hots: (torch.Tensor) Represent the board in one hot encoding. (batch_size, 81, 9)
		targets: (torch.Tensor) All of the value scores of the boards (batch_size, )
		Returns:
			the loss as a scalar
		'''
		loss = None
		b, t = one_hots.shape[:2]
		assert t == self.block_size

		# forward the GPT model
		token_embeddings = self.one_hot_emb(one_hots) # each idx maps to a learned vector
		position_embeddings = self.pos_emb[:, :t, :] # each pos maps to a learned vector
		x = self.drop(token_embeddings + position_embeddings)
		# collect the attention matrices and last layer predicted logits
		atts = []
		for _ in range(self.n_recur):
			for block in self.blocks:
				x, att = block(x) # (batch, 81, 128) (batch, num_heads, 81, 81)
				atts.append(att)
		tail_output = self.ln_f(x)
		tail_output = tail_output.view(tail_output.size(0), -1) #(batch, 81*emb_dim)
		value_hat = self.head(tail_output).squeeze() #(batch,)
		# compute losses
		if targets is not None:
			loss = nn.MSELoss()(value_hat, targets)
		else:
			raise ValueError()

		return loss