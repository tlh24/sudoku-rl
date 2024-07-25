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
	def forward(ctx, input, std, activ, n_enable):
		batch_size = input.shape[0]
		if n_enable is None: 
			n_enable = 0
		else: 
			n_enable = n_enable // 1000 + 1
		
		ac = torch.squeeze(activ)
		# ac[n_enable:] = 10 # turn the other units off.
		ac = torch.exp(-5.0*ac)
		s = torch.sum(ac)
		ac[0] = s*4000 # controls the probability of adding a new unit
		# seems to be a sensitive hyper-parameter: might want to change per layer?
		# unit zero is hence never enabled.
		# this causes scaling problems... meh.
		r = torch.multinomial(ac, batch_size, replacement=True) # sample batch_size rows to activate, based on the probability distribution 'ac'.
		i = torch.arange(batch_size)
		x = input
		x[i,0,r] = x[i,0,r] + (std * (r > 0)) # * (torch.randint(2,(1,), device=x.device) * 2 - 1)
		# mask off the zeroth unit; vary the sign of the activation.
		return x

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output, None, None, None # grad for: input, std, activ
		
class StraightThroughNormal(nn.Module):
	def __init__(self,n):
		super(StraightThroughNormal, self).__init__()
		self.register_buffer('activ', torch.zeros(1,n))

	def forward(self, x, std, n_enable=None):
		self.activ = 0.97 * self.activ + 0.03 * torch.mean(torch.abs(x), dim=0) # or x^2 ?
		x = STNFunction.apply(x, std, self.activ.detach(), n_enable)
		return x

class NetSimp2(nn.Module):
	# the absolute simplest network that can be zero-initialized
	def __init__(self, init_zeros:bool):
		super(NetSimp2, self).__init__()
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
		output = F.log_softmax(y, dim=1) # necessary with nll_loss
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
	# a simple MLP with 3 layers (2 hidden, one output)
	def __init__(self, init_zeros:bool):
		super(NetSimp3, self).__init__()
		self.init_zeros = init_zeros
		self.fc1 = nn.Linear(784, 512)
		self.stn1 = StraightThroughNormal(512)
		self.fc2 = nn.Linear(512, 128)
		self.stn2 = StraightThroughNormal(128)
		self.fc3 = nn.Linear(128, 10)
		self.gelu = QuickGELU()
		if init_zeros: 
			torch.nn.init.zeros_(self.fc1.weight)
			torch.nn.init.zeros_(self.fc1.bias)
			torch.nn.init.zeros_(self.fc2.weight)
			torch.nn.init.zeros_(self.fc2.bias)
			torch.nn.init.zeros_(self.fc3.weight)
			torch.nn.init.zeros_(self.fc3.bias)
		
	def forward(self, x, n_enable): 
		x = torch.reshape(x, (-1, 1, 784))
		x = self.fc1(x)
		if self.init_zeros:
			x2 = self.stn1(x, 0.001, n_enable)
		else:
			x2 = x
		x2 = self.gelu(x2)
		x2 = self.fc2(x2)
		if self.init_zeros:
			x3 = self.stn2(x2, 0.001, n_enable)
		else:
			x3 = x2
		x3 = self.gelu(x3)
		x3 = self.fc3(x3)
		y = torch.squeeze(x3)
		output = F.log_softmax(y, dim=1) # necessary with nll_loss
		sum_activ = torch.sum(x**2) + torch.sum(x2**2)
		return output, sum_activ
		
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

class NetSimp4(nn.Module):
	# a simple MLP with 3 layers
	def __init__(self, init_zeros:bool):
		super(NetSimp4, self).__init__()
		self.init_zeros = init_zeros
		self.fc1 = nn.Linear(784, 1500)
		self.stn1 = StraightThroughNormal(1500)
		self.fc2 = nn.Linear(1500, 384)
		self.stn2 = StraightThroughNormal(384)
		self.fc3 = nn.Linear(384, 96)
		self.stn3 = StraightThroughNormal(96)
		self.fc4 = nn.Linear(96, 10)
		self.gelu = QuickGELU()
		if init_zeros:
			torch.nn.init.zeros_(self.fc1.weight)
			torch.nn.init.zeros_(self.fc1.bias)
			torch.nn.init.zeros_(self.fc2.weight)
			torch.nn.init.zeros_(self.fc2.bias)
			torch.nn.init.zeros_(self.fc3.weight)
			torch.nn.init.zeros_(self.fc3.bias)
			torch.nn.init.zeros_(self.fc4.weight)
			torch.nn.init.zeros_(self.fc4.bias)

	def forward(self, x, n_enable):
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
		if self.init_zeros:
			x = self.stn3(x, 0.01)
		x = self.gelu(x)
		x = self.fc4(x)
		y = torch.squeeze(x)
		output = F.log_softmax(y, dim=1) # necessarry with nll_loss
		return output, torch.tensor(0.0)

class NetSimpAE(nn.Module):
	# a simple MLP with 3 layers
	def __init__(self, init_zeros:bool):
		super(NetSimpAE, self).__init__()
		self.init_zeros = init_zeros
		self.fc1 = nn.Linear(784, 256)
		self.stn1 = StraightThroughNormal(256)
		self.fc2 = nn.Linear(256, 48)
		self.stn2 = StraightThroughNormal(48)
		self.fc3 = nn.Linear(48, 256)
		self.stn3 = StraightThroughNormal(256)
		self.fc4 = nn.Linear(256, 784)

		self.fco1 = nn.Linear(48, 20)
		self.fco2 = nn.Linear(20, 10)
		self.gelu = QuickGELU()
		if init_zeros:
			torch.nn.init.zeros_(self.fc1.weight)
			torch.nn.init.zeros_(self.fc1.bias)
			torch.nn.init.zeros_(self.fc2.weight)
			torch.nn.init.zeros_(self.fc2.bias)
			torch.nn.init.zeros_(self.fc3.weight)
			torch.nn.init.zeros_(self.fc3.bias)
			torch.nn.init.zeros_(self.fc4.weight)
			torch.nn.init.zeros_(self.fc4.bias)

			# torch.nn.init.zeros_(self.fco1.weight)
			# torch.nn.init.zeros_(self.fco1.bias)
			# torch.nn.init.zeros_(self.fco2.weight)
			# torch.nn.init.zeros_(self.fco2.bias)

	def forwardAE(self, x):
		x = torch.reshape(x, (-1, 1, 784))
		x = self.fc1(x)
		if self.init_zeros:
			x = self.stn1(x, 0.01)
		x = self.gelu(x)
		x = self.fc2(x)
		if self.init_zeros:
			x = self.stn2(x, 0.01)
		x = self.gelu(x)
		y = x
		x = self.fc3(x)
		if self.init_zeros:
			x = self.stn3(x, 0.01)
		x = self.gelu(x)
		x = self.fc4(x)
		x = x.reshape(-1,28,28)
		return x

	def forwardClass(self, x):
		with torch.no_grad():
			x = torch.reshape(x, (-1, 1, 784))
			x = self.fc1(x)
			x = self.gelu(x)
			x = self.fc2(x)
			x = self.gelu(x)
			y = x
		y = self.fco1(y)
		y = self.gelu(y)
		y = self.fco2(y)
		y = y.squeeze()
		output = F.log_softmax(y, dim=1) # necessarry with nll_loss
		return output.squeeze()



def train(args, model, device, train_im, train_lab, optimizer, uu, mode, fd_losslog):
	indx = torch.randperm(train_lab.shape[0])
	indx = indx[:args.batch_size]
	im = train_im[indx,:,:]
	lab = train_lab[indx]
	images = im.to(device)
	labels = lab.to(device)
	
	threshold = args.num_iters # pdgd seems to hurt here! HUH
	# if args.z: 
	threshold = 0

	if uu < threshold:
		optimizer[0].zero_grad()
		if mode == 0:
			x = model.forwardAE(images)
			loss = torch.sum((x - images)**2) / 70
		if mode == 1:
			output = model.forwardClass(images)
			loss = F.nll_loss(output, labels)
		if mode == 2: 
			output,sum_activ = model.forward(images, uu )
			loss = F.nll_loss(output, labels) # + 1e-6 * sum_activ
		loss.backward()
		optimizer[0].step()
	else:
		def closure():
			output,_ = model(images, uu)
			loss = F.nll_loss(output, labels) + sum( \
					[torch.sum(1e-3 * torch.rand_like(param) * param * param) for param in model.parameters()])
			return loss
		loss = optimizer[1].step(closure)

	lloss = loss.detach().cpu().item()
	fd_losslog.write(f'{uu}\t{lloss}\t0.0\n')
	fd_losslog.flush()

	if args.g:
		if uu % 10 == 9:
			print(mode, lloss)
	# else:
	# 	if uu % 120 == 119:
	# 		print(".", end="", flush=True)


def test(args, model, device, test_im, test_lab, fd_results):
	with torch.no_grad():
		test_im = test_im.to(device)
		test_lab = test_lab.to(device)
		if args.ae: 
			output = model.forwardClass(test_im)
		else: 
			output,_ = model.forward(test_im, None)
		test_loss = F.nll_loss(output, test_lab, reduction='sum').item()
		pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
		correct = pred.eq(test_lab.view_as(pred)).sum().item()

	test_loss /= test_im.shape[0]

	print('Test set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
		test_loss, correct, test_im.shape[0],
		100. * correct / test_im.shape[0]))
	fd_results.write('{}\t{}\t{}\t{:.3f}\t'.format(args.z, args.train_size, args.num_iters, correct / test_im.shape[0]))


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
	parser.add_argument('-c', type=int, default=0, metavar='N',
							help='which CUDA device to use')
	parser.add_argument('-z', action='store_true', default=False,
							help='Zero init')
	parser.add_argument('--adamw', action='store_true', default=False,
							help='use AdamW')
	parser.add_argument('--adagrad', action='store_true', default=False,
							help='use AdaGrad')
	parser.add_argument('-g', action='store_true', default=False,
							help='debug / print loss')
	parser.add_argument('--ae', action='store_true', default=False,
							help='use autoencoder')
	parser.add_argument('-p', action='store_true', default=False,
							help='plot weight histograms')
	# parser.add_argument('--seed', type=int, default=1, metavar='S',
	# 						help='random seed (default: 1)')

	args = parser.parse_args()

	# torch.manual_seed(args.seed)

	device = torch.device(f"cuda:{args.c}")

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
		
	fd_results = open('mnist_zeroinit.txt', 'a')
	fd_losslog = open('../losslog.txt', 'w')
	
	if args.p: 
		plot_rows = 10
		plot_cols = 3
		figsize = (16, 24)
		fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)

	for repeat in range(10): 
			
		if args.ae: 
			model = NetSimpAE(init_zeros = args.z).to(device)
		else: 
			model = NetSimp4(init_zeros = args.z).to(device)
			
		optimizer = []
	
		if args.adamw: 
			optimizer.append(optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True, weight_decay=0.01))
		if args.adagrad:
			optimizer.append( optim.Adagrad(model.parameters(), lr=0.015, weight_decay=0.01) )

		optimizer.append( psgd.LRA(model.parameters(),lr_params=0.01,\
			lr_preconditioner=0.01, momentum=0.9,\
			preconditioner_update_probability=0.1, \
			exact_hessian_vector_product=False, \
			rank_of_approximation=10, grad_clip_max_norm=5.0) )

		if args.ae: 
			for uu in range(args.num_iters):
				train(args, model, device, images, torch.zeros_like(labels), optimizer1, uu, 0, fd_losslog)

		shuffl = torch.randperm(70000)
		train_im = images[shuffl[0:args.train_size],:,:]
		train_lab = labels[shuffl[0:args.train_size]]
		test_im = images[shuffl[args.train_size:],:,:]
		test_lab = labels[shuffl[args.train_size:]]

		train_lab = train_lab.type(torch.LongTensor)
		test_lab = test_lab.type(torch.LongTensor)
		
		if args.ae: 
			for uu in range(args.num_iters):
				train(args, model, device, train_im, train_lab, optimizer, uu, 1, fd_losslog)
		else: 
			for uu in range(args.num_iters // 10 * repeat):
				train(args, model, device, train_im, train_lab, optimizer, uu, 2, fd_losslog)


		test(args, model, device, test_im, test_lab, fd_results)
		
		w = [model.fc1.weight.detach().cpu().numpy(), \
					model.fc2.weight.detach().cpu().numpy(), \
					model.fc3.weight.detach().cpu().numpy()]
		
		sparsity = np.zeros((3,))
		num_nonzero = 0
		for j in range(3): 
			x = w[j].flatten()
			nonsparse = np.count_nonzero(abs(x) > 1e-5)
			num_nonzero = num_nonzero + nonsparse
			sparsity[j] = int(1000 * (1 - nonsparse/x.size)) / 1000.0
		print(f" sparsity:{sparsity[0]},{sparsity[1]},{sparsity[2]};tot nonzero param:{num_nonzero}")
		fd_results.write(f"{sparsity[0]}\t{sparsity[1]}\t{sparsity[2]}\n")
		fd_results.flush()
		
		
		if args.p: 
			
			for j in range(3): 
				x = w[j].flatten()
				if args.z:
					rnge = (-0.5, 0.5)
				else:
					rnge = (-0.2, 0.2)
				axs[repeat,j].hist(x, 400, rnge)
				nonsparse = np.count_nonzero(x)
				axs[repeat,j].set_title(f'histogram weight matrix {j}; sparsity:{sparsity[j]}')
				axs[repeat,j].set_yscale('log', nonpositive='clip')
				# axs[0,j].yscale('log')
				
			# axs[1,0].plot(model.stn1.activ.detach().cpu().squeeze().numpy())
			# axs[1,0].set_title(f'H1 unit fading memory average activity')
			# axs[1,1].plot(model.stn2.activ.detach().cpu().squeeze().numpy())
			# axs[1,1].set_title(f'H2 unit fading memory average activity')
			
	plt.show()

	fd_results.close()

if __name__ == '__main__':
    main()
