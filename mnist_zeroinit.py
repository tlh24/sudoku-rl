from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pdb
import matplotlib.pyplot as plt
import numpy as np
import math 
import psgd

# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.conv1 = nn.Conv2d(1, 32, 3, 1)
# 		self.conv2 = nn.Conv2d(32, 64, 3, 1)
# 		self.dropout1 = nn.Dropout(0.25)
# 		self.dropout2 = nn.Dropout(0.5)
# 		self.fc1 = nn.Linear(9216, 128)
# 		self.fc2 = nn.Linear(128, 10)
#
# 	def forward(self, x):
# 		x = self.conv1(x)
# 		x = F.relu(x)
# 		x = self.conv2(x)
# 		x = F.relu(x)
# 		x = F.max_pool2d(x, 2)
# 		x = self.dropout1(x)
# 		x = torch.flatten(x, 1)
# 		x = self.fc1(x)
# 		x = F.relu(x)
# 		x = self.dropout2(x)
# 		x = self.fc2(x)
# 		output = F.log_softmax(x, dim=1)
# 		return output


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
	  
class STNFunction(torch.autograd.Function):
	# see https://www.kaggle.com/code/peggy1502/learning-pytorch-2-new-autograd-functions/notebook
	# and https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
	@staticmethod
	def forward(ctx, input, std, activ):
		batch_size = input.shape[0]
		ac = torch.squeeze(activ)
		ac = torch.exp(-5.0*ac)
		s = torch.sum(ac)
		ac[0] = s*25 # controls the probability of adding a new unit
		# unit zero is hence never enabled.
		# this causes scaling problems... meh.
		r = torch.multinomial(ac, batch_size, replacement=True) # sample 1 row to activate, based on the probability distribution 'ac'.
		i = torch.arange(batch_size)
		x = input
		x[i,0,r] = x[i,0,r] + (std * (r > 0))
		return x

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output, None, None # grad for: input, std, activ
		
class StraightThroughNormal(nn.Module):
	def __init__(self,n):
		super(StraightThroughNormal, self).__init__()
		self.register_buffer('activ', torch.zeros(1,n))

	def forward(self, x, std):
		self.activ = 0.97 * self.activ + 0.03 * torch.mean(torch.abs(x), dim=0) # or x^2 ?
		x = STNFunction.apply(x, std, self.activ.detach())
		return x

class NetSimp(nn.Module): 
	# the absolute simplest network that can be zero-initialized
	def __init__(self, init_zeros:bool):
		super(NetSimp, self).__init__()
		self.init_zeros = init_zeros
		self.fc1 = nn.Linear(784, 250)
		self.stn = StraightThroughNormal(250)
		self.gelu = QuickGELU()
		self.fc2 = nn.Linear(250, 10)
		if init_zeros: 
			torch.nn.init.zeros_(self.fc1.weight)
			torch.nn.init.zeros_(self.fc1.bias)
			torch.nn.init.zeros_(self.fc2.weight)
			torch.nn.init.zeros_(self.fc2.bias)
		
	def forward(self, x): 
		x = torch.reshape(x, (-1, 1, 784))
		x = self.fc1(x)
		if self.init_zeros:
			x = self.stn(x, 0.001) # algorithm is not sensitive to this parameter 
		x = self.gelu(x)
		x = self.fc2(x)
		y = torch.squeeze(x)
		output = F.log_softmax(y, dim=1) # necessarry with nll_loss
		return output
		
	def hidden(self, x):
		x = torch.reshape(x, (-1, 1, 784))
		x = self.fc1(x)
		return x # self.gelu(x)

	def load_checkpoint(self, path:str=None):
		if path is None:
			path = "mnist_2layer.pth"
		self.load_state_dict(torch.load(path))

	def save_checkpoint(self, path:str=None):
		if path is None:
			path = "mnist_2layer.pth"
		torch.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")
		
class NetSimp3(nn.Module): 
	# a simple MLP with 3 layers
	def __init__(self, init_zeros:bool):
		super(NetSimp3, self).__init__()
		self.init_zeros = init_zeros
		self.fc1 = nn.Linear(784, 1500)
		self.stn1 = StraightThroughNormal(1500)
		self.fc2 = nn.Linear(1500, 250)
		self.stn2 = StraightThroughNormal(250)
		self.fc3 = nn.Linear(250, 10)
		self.gelu = QuickGELU()
		if init_zeros: 
			torch.nn.init.zeros_(self.fc1.weight)
			torch.nn.init.zeros_(self.fc1.bias)
			torch.nn.init.zeros_(self.fc2.weight)
			torch.nn.init.zeros_(self.fc2.bias)
			torch.nn.init.zeros_(self.fc3.weight)
			torch.nn.init.zeros_(self.fc3.bias)
		
	def forward(self, x): 
		x = torch.reshape(x, (-1, 1, 784))
		x = self.fc1(x)
		if self.init_zeros:
			x = self.stn1(x, 0.01)
		x = self.gelu(x)
		x = self.fc2(x)
		if self.init_zeros:
			x = self.stn2(x, 0.01)
		x = self.gelu(x)
		x = self.fc3(x)
		y = torch.squeeze(x)
		output = F.log_softmax(y, dim=1) # necessarry with nll_loss
		return output
		
	def hidden(self, x):
		x = torch.reshape(x, (-1, 1, 784))
		x = self.fc1(x)
		x = self.gelu(x)
		x = self.fc2(x)
		return self.gelu(x)

	def load_checkpoint(self, path:str=None):
		if path is None:
			path = "mnist_3layer.pth"
		self.load_state_dict(torch.load(path))

	def save_checkpoint(self, path:str=None):
		if path is None:
			path = "mnist_3layer.pth"
		torch.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")


def train(args, model, device, train_im, train_lab, optimizer, uu):
	indx = torch.randperm(train_lab.shape[0])
	indx = indx[:args.batch_size]
	im = train_im[indx,:,:]
	lab = train_lab[indx]
	images = im.to(device)
	labels = lab.to(device)

	def closure():
		output = model(images)
		loss = F.nll_loss(output, labels)
		return loss

	loss = optimizer.step(closure)
	if uu % 10 == 9:
		lloss = loss.detach().cpu().item()
		print(lloss)


def test(model, device, test_im, test_lab):
	with torch.no_grad():
		test_im = test_im.to(device)
		test_lab = test_lab.to(device)
		output = model(test_im)
		test_loss = F.nll_loss(output, test_lab, reduction='sum').item()
		pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
		correct = pred.eq(test_lab.view_as(pred)).sum().item()

	test_loss /= test_im.shape[0]

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, test_im.shape[0],
		100. * correct / test_im.shape[0]))


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='testing MNIST for generalization w & wo zero-init')
	parser.add_argument('--train-size', type=int, default=128, metavar='N',
							help='input data size for training (default: 128)')
	parser.add_argument('--num-iters', type=int, default=15000, metavar='N',
							help='number of training iterations (default: 15000)')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
							help='input data size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=5, metavar='N',
							help='number of epochs to train (default: 5)')
	parser.add_argument('--cuda-device', type=int, default=0, metavar='N',
							help='which CUDA device to use')
	parser.add_argument('-z', action='store_true', default=False,
							help='Zero init')
	# parser.add_argument('--seed', type=int, default=1, metavar='S',
	# 						help='random seed (default: 1)')

	args = parser.parse_args()

	# torch.manual_seed(args.seed)

	device = torch.device(f"cuda:{args.cuda_device}")

	train_kwargs = {'batch_size': args.batch_size}
	test_kwargs = {'batch_size': args.batch_size}
	cuda_kwargs = {'num_workers': 1,
						'pin_memory': True,
						'shuffle': True}
	train_kwargs.update(cuda_kwargs)
	test_kwargs.update(cuda_kwargs)

	transform=transforms.Compose([
		transforms.ToTensor(),
		# transforms.RandomAffine(2.5, (0.1, 0.1), (0.95, 1.05)), 
		transforms.Normalize((0.1307,), (0.3081,))
		])

	dataset1 = datasets.MNIST('../data', train=True, download=True,
							transform=transform)
	dataset2 = datasets.MNIST('../data', train=False,
							transform=transform)

	images = torch.zeros(70000, 28, 28)
	labels = torch.zeros(70000)
	for i in range(len(dataset1)):
		im, lab = dataset1[i]
		images[i,:,:] = im[0,:,:]
		labels[i] = lab
	for i in range(len(dataset2)):
		im, lab = dataset2[i]
		j = i + 60000
		images[j,:,:] = im[0,:,:]
		labels[j] = lab

	shuffl = torch.randperm(70000)
	train_im = images[shuffl[0:args.train_size],:,:]
	train_lab = labels[shuffl[0:args.train_size]]
	test_im = images[shuffl[args.train_size:],:,:]
	test_lab = labels[shuffl[args.train_size:]]

	train_lab = train_lab.type(torch.LongTensor)
	test_lab = test_lab.type(torch.LongTensor)

	model = NetSimp3(init_zeros = args.z).to(device)

	optimizer = psgd.LRA(model.parameters(),lr_params=0.01,\
			lr_preconditioner=0.01, momentum=0.9,\
			preconditioner_update_probability=0.1, \
			exact_hessian_vector_product=False, \
			rank_of_approximation=10, grad_clip_max_norm=5.0)

	for uu in range(args.num_iters):
		train(args, model, device, train_im, train_lab, optimizer, uu)

	test(model, device, test_im, test_lab)


if __name__ == '__main__':
    main()
