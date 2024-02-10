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
		ac[0] = s*999 # controls the probability of adding a new unit
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
	# the absolute simplest network
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
	# the absolute simplest network
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

class NetDenoise(nn.Module): 
	def __init__(self):
		super(NetDenoise, self).__init__()
		H = 250
		self.fc1 = nn.Linear(H+1, 125)
		self.fc2 = nn.Linear(126, 64)
		self.fc3 = nn.Linear(65, 125)
		self.fc4 = nn.Linear(126, H)
		self.gelu = QuickGELU()
		
	def forward(self, x, t): 
		# t is the noise std. dev, as in diffusion models. 
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
			path = "denoise.pth"
		self.load_state_dict(torch.load(path))

	def save_checkpoint(self, path:str=None):
		if path is None:
			path = "denoise.pth"
		torch.save(self.state_dict(), path)
		print(f"saved checkpoint to {path}")

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
							help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
							help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=5, metavar='N',
							help='number of epochs to train (default: 5)')
	parser.add_argument('--lr', type=float, default=0.006, metavar='LR',
							help='learning rate (default: 0.006)')
	parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
							help='Learning rate step gamma (default: 0.95)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
							help='disables CUDA training')
	parser.add_argument('--dry-run', action='store_true', default=False,
							help='quickly check a single pass')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
							help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
							help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
							help='For Saving the current Model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	if use_cuda:
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	train_kwargs = {'batch_size': args.batch_size}
	batch_size = args.batch_size
	test_kwargs = {'batch_size': args.test_batch_size}
	if use_cuda:
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
	train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
	test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

	model = NetSimp(init_zeros = True).to(device) 
	denoise = NetDenoise().to(device)
	try: 
		model.load_checkpoint()
	except:
		print('could not load mnist model weights')
	try: 
		denoise.load_checkpoint()
	except:
		print('could not load denoise model weights')

	# optimizer = optim.AdamW(model.parameters(), lr=1e-3)
	optimizer = optim.Adagrad(model.parameters(), lr=0.015, weight_decay=0.01)
	# adagrad has the strongest regularization
	# optimizer = optim.Adadelta(model.parameters(), lr=0.05, weight_decay=0.01)
	# optimizer = optim.RMSprop(model.parameters(), lr=0.005, weight_decay=0.01) # really bad
	# optimizer = optim.SGD(model.parameters(), lr=0.02, weight_decay=0.01) # quite slow, but works.
	# Adagrad works well for this simple problem.
	# AdamW has (surprisingly) worse sparsity.
	# might want to switch optimizer?
	
	denoiseopt = optim.AdamW(denoise.parameters(), lr=1e-3, weight_decay = 5e-2)

	if True and args.epochs > 0:
		scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
		for epoch in range(1, args.epochs + 1):
			train(args, model, device, train_loader, optimizer, epoch)
			test(model, device, test_loader)
			scheduler.step()

		model.save_checkpoint()

		plot_rows = 3
		plot_cols = 2
		figsize = (16, 8)
		fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)

		w = [model.fc1.weight.detach().cpu().numpy(), \
			model.fc2.weight.detach().cpu().numpy() ]
		
		pcm = axs[0,1].imshow(w[1])
		fig.colorbar(pcm, ax=axs[0,1])
		for j in range(2): 
			pcm = axs[0,j].imshow(w[j])
			fig.colorbar(pcm, ax=axs[0,j])
			axs[0,j].set_title(f'weight matrix {j+1}')
			axs[j+1,0].plot(np.var(w[j], 1))
			axs[j+1,0].set_title(f'variances of weight matrix {j+1} along output dim')
			axs[j+1,1].hist(np.var(w[j], 1), 140)
			axs[j+1,1].set_title(f'histogram of variances of weight matrix {j+1} along output dim')

		plt.show()
		
	if True: 
		# record the hidden unit activations.
		# if we drive the hidden units with a denoiser, 
		# can you also re-create the digit? 
		H = 250
		indata = torch.zeros(60000, 28, 28).to(device)
		hidden = torch.zeros(60000, H).to(device)
		for i, (data, target) in enumerate(train_loader):
			data = data.to(device)
			h = model.hidden(data)
			hidden[i*batch_size:(i+1)*batch_size, :] = torch.squeeze(h)
			indata[i*batch_size:(i+1)*batch_size, :, :] = torch.squeeze(data)
		print('done generating hidden')
		
		N = 40000
		losses = np.zeros((N,))
		
		for u in range(N): 
			with torch.no_grad():
				i = torch.randint(60000, (batch_size,)).to(device)
				t = torch.rand(batch_size).to(device)
				tx = t.unsqueeze(-1).expand((-1, H))
				x = hidden[i,:]
				xn = x + torch.randn(batch_size, H).to(device) * tx * 2.5
			denoiseopt.zero_grad()
			y = denoise.forward(xn,t)
			loss = torch.sum((y - x)**2)
			loss.backward()
			denoiseopt.step()
			losses[u] = loss.cpu().detach().item()
			# print(losses[u])
			# if u % 10000 == 9999: 
			# 	plt.plot(x[0,:].cpu().numpy(), 'k', label='x')
			# 	plt.plot(xn[0,:].cpu().numpy(), 'b', label='x + noise')
			# 	plt.plot(y[0,:].cpu().detach().numpy(), 'r', label='denoised')
			# 	plt.legend()
			# 	plt.show()
			
		denoise.save_checkpoint()
		# plt.plot(np.log(losses))
		# plt.show()
		
		print('checking inversion')
		xnp = np.random.normal(0, 0.1, (10,28,28))
		x = torch.tensor(xnp, requires_grad=True, device=device, dtype=torch.float)
		with torch.no_grad():
			indx = torch.randint(60000, (10,)).to(device)
			hdn = hidden[indx, :] # no grad here
		
		N = 2000
		losses = np.zeros((N,))
		
		for u in range(N): 
			model.zero_grad()
			x.grad = None
			h = model.hidden(x)
			h = torch.squeeze(h)
			with torch.no_grad():
				z = torch.randn_like(hdn) * 0.2
				hdnz = hdn+z # noisy target? seems not to make much difference.
			loss = torch.sum((h - hdnz)**2) # yes, this is just a matrix... 
			loss.backward() # this interacts in some weird way .. ? 
			losses[u] = loss.cpu().detach().item()
			with torch.no_grad():
				torch.nn.utils.clip_grad_norm_([x], 1.0)
				x -= x.grad * 0.05
			if False: 
				axs[0].cla()
				axs[0].plot(h[0,:].cpu().numpy(), 'b')
				axs[0].plot(hdn[0,:].cpu().numpy(), 'k')
				axs[1].cla()
				axs[1].imshow(x[0,:,:].cpu().detach().numpy())
				fig.tight_layout()
				fig.canvas.draw()
				fig.canvas.flush_events()
				
		fig,axs = plt.subplots(3,6, figsize=(18,10))
		axs[0,0].plot(losses)
		axs[0,0].set_title('hidden losses w fixed target')
		for j in range(5): 
			c = j+1
			axs[0,c].plot(hdn[j,:].cpu().detach().numpy(), 'b')
			axs[0,c].plot(h[j,:].cpu().detach().numpy(), 'r')
			q = h - hdn
			axs[0,c].plot(q[j,:].cpu().detach().numpy(), 'k')
			axs[0,c].set_title(f'blue = target hidden; red = denoised; black = residual')
			im = axs[1,c].imshow(x[j,:,:].cpu().detach().numpy())
			plt.colorbar(im, ax=axs[1,c])
			axs[1,c].set_title(f'resultant image {j}')
			# show the actual image as well.
			k = indx[j]
			im = axs[2,c].imshow(indata[k,:,:].cpu().detach().numpy())
			plt.colorbar(im, ax=axs[2,c])
			axs[2,c].set_title(f'original image {j}')
		plt.show()

	# need to propagate activity backwards, see what the image looks like.
	# (this of course makes me think of a diffusion model, the current champion of conditional image generation.
	N = 8000
	losses = np.zeros((N,2))
	xnp = np.random.normal(0, 0.1, (10,28,28))
	x = torch.tensor(xnp, requires_grad=True, device=device, dtype=torch.float)
	y = torch.zeros((10,), device=device, dtype=torch.long)
	for j in range(10): 
		y[j] = j
		
	plt.ion()
	fig, axs = plt.subplots(1, 2, figsize=(10, 5))
	for i in range(N):
		model.zero_grad() # does nothing, we're not taking grad wrt parameters.
		x.grad = None
		yp = model(x)
		loss = F.nll_loss(yp, y)
		loss.backward()
		losses[i,0] = loss.cpu().detach().item()
		# print(losses[i])
		with torch.no_grad():
			torch.nn.utils.clip_grad_norm_([x], 0.4)
			x -= x.grad * 0.1
			x -= x * 0.0001
		
		# now, do the same for the hidden layer denoise.
		if True:
			x.grad = None
			h = model.hidden(x)
			h = torch.squeeze(h)
			with torch.no_grad(): 
				t = torch.ones(10) * (N - i) / (N+2.0)
				t = t.to(device)
				z = torch.randn_like(h) * 0.75 * ((N - i) / (N+2.0)) # anneal
				hdn = denoise(h+z,t)
				if i % 100 == 0: 
					axs[0].cla()
					axs[0].plot(h[0,:].cpu().numpy(), 'b')
					axs[0].plot(hdn[0,:].cpu().numpy(), 'k')
					q = h - hdn - 1.0
					axs[0].plot(q[0,:].cpu().numpy(), 'r')
					axs[0].set_title('black = hidden; blue = hidden denoised, red = residual')
					axs[1].cla()
					axs[1].imshow(x[0,:,:].cpu().detach().numpy())
					fig.tight_layout()
					fig.canvas.draw()
					fig.canvas.flush_events()
			loss = torch.sum((h - hdn)**2) # yes, this is just a matrix... 
			loss.backward() # this interacts in some weird way .. ? 
			losses[i,1] = loss.cpu().detach().item()
			with torch.no_grad():
				torch.nn.utils.clip_grad_norm_([x], 0.1)
				x -= x.grad * 0.035
		
		if i == N-1:
			plt.ioff()
			fig,axs = plt.subplots(2,6, figsize=(18,9))
			axs[0,0].plot(losses[:,0])
			axs[0,0].set_title('output y losses')
			axs[1,0].plot(losses[:,1])
			axs[1,0].set_title('hidden losses')
			for j in range(10): 
				r = j // 5
				c = j % 5 + 1
				im = axs[r,c].imshow(x[j,:,:].cpu().detach().numpy())
				plt.colorbar(im, ax=axs[r,c])
				axs[r,c].set_title(f'resultant image {j}')
			plt.show()


if __name__ == '__main__':
    main()
