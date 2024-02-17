import torch
from torch import nn
from graph_transformer import QuickGELU 

class NetDenoise(nn.Module): 
	def __init__(self, H):
		super(NetDenoise, self).__init__()
		self.h = H
		self.fc1 = nn.Linear(H+1, H//2)
		self.fc2 = nn.Linear(H//2+1, H//4)
		self.fc3 = nn.Linear(H//4+1, H//2)
		self.fc4 = nn.Linear(H//2+1, H)
		self.gelu = QuickGELU()
		
	def forward(self, x, t): 
		# t = temp, varies from 0 (no noise) to 1 (pure noise). 
		t = t.unsqueeze(-1)
		x = self.fc1(torch.cat((x,t), 1))
		x = self.gelu(x)
		x = self.fc2(torch.cat((x,t), 1))
		x = self.gelu(x)
		x = self.fc3(torch.cat((x,t), 1))
		x = self.gelu(x)
		x = self.fc4(torch.cat((x,t), 1))
		return x
		
	def load_checkpoint(self, path:str=None):
		if path is None:
			path = f"denoise_{self.h}.pth"
		self.load_state_dict(torch.load(path))

	def save_checkpoint(self, path:str=None):
		if path is None:
			path = f"denoise_{self.h}.pth"
		torch.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")
