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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
	  
class STNFunction(torch.autograd.Function):
	# see https://www.kaggle.com/code/peggy1502/learning-pytorch-2-new-autograd-functions/notebook
	# and https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
	@staticmethod
	def forward(ctx, input, std, activ):
		# don't add noise to active tensors
		x = input + torch.randn_like(input) * (std * torch.exp(-4.0*activ))
		ctx.save_for_backward(x)
		return x

	@staticmethod
	def backward(ctx, grad_output):
		x, = ctx.saved_tensors
		return F.hardtanh(grad_output * x), None, None # clip gradients? 
		# note: no need to gate the noise. 
		# if the error is zero, grad_output will be zero as well.
		
class StraightThroughNormal(nn.Module):
	def __init__(self):
		super(StraightThroughNormal, self).__init__()
		self.register_buffer('activ', torch.randn(1))

	def forward(self, x, std):
		if self.activ.shape != x.shape: 
			self.activ = torch.zeros_like(x)
		self.activ = 0.95 * self.activ + 0.05 * torch.abs(x) # or x^2 ? 
		x = STNFunction.apply(x, std, self.activ.detach())
		return x

class NetSimp(nn.Module): 
	# the absolute simplest network
	def __init__(self, init_zeros:bool):
		super(NetSimp, self).__init__()
		self.fc1 = nn.Linear(784, 2500)
		self.stn = StraightThroughNormal()
		self.fc2 = nn.Linear(2500, 10)
		if init_zeros: 
			torch.nn.init.zeros_(self.fc1.weight) # bias starts at zero by default
			torch.nn.init.zeros_(self.fc2.weight)
		
	def forward(self, x): 
		x = torch.reshape(x, (-1, 1, 784))
		x = self.fc1(x)
		x = self.stn(x, 0.01 / 6) # negative values don't matter because of the relu!
		x = F.relu(x)
		x = self.fc2(x)
		y = torch.squeeze(x)
		output = F.log_softmax(y, dim=1) # necessarry with nll_loss
		return output


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
	parser.add_argument('--epochs', type=int, default=4, metavar='N',
							help='number of epochs to train (default: 5)')
	parser.add_argument('--lr', type=float, default=0.006, metavar='LR',
							help='learning rate (default: 0.06)')
	parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
							help='Learning rate step gamma (default: 0.95)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
							help='disables CUDA training')
	parser.add_argument('--no-mps', action='store_true', default=False,
							help='disables macOS GPU training')
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
	use_mps = not args.no_mps and torch.backends.mps.is_available()

	torch.manual_seed(args.seed)

	if use_cuda:
		device = torch.device("cuda")
	elif use_mps:
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	train_kwargs = {'batch_size': args.batch_size}
	test_kwargs = {'batch_size': args.test_batch_size}
	if use_cuda:
		cuda_kwargs = {'num_workers': 1,
							'pin_memory': True,
							'shuffle': True}
		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])
	dataset1 = datasets.MNIST('../data', train=True, download=True,
							transform=transform)
	dataset2 = datasets.MNIST('../data', train=False,
							transform=transform)
	train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
	test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

	model = NetSimp(init_zeros = True).to(device)
	optimizer = optim.AdamW(model.parameters(), lr=args.lr)

	scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)
		scheduler.step()

	if args.save_model:
		torch.save(model.state_dict(), "mnist_cnn.pt")
	
	plot_rows = 2
	plot_cols = 2
	figsize = (16, 8)
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	
	w1 = model.fc1.weight.detach().cpu().numpy()
	w2 = model.fc2.weight.detach().cpu().numpy()
	pcm = axs[0,0].imshow(w1)
	fig.colorbar(pcm, ax=axs[0,0])
	axs[1,0].plot(np.var(w1, 1))
	axs[1,1].hist(np.var(w1, 1), 40)
	axs[1,1].set_title('histogram of variances of weight matrix 1 along output dim')
	
	pcm = axs[0,1].imshow(w2)
	fig.colorbar(pcm, ax=axs[0,1])
	
	plt.show()


if __name__ == '__main__':
    main()
