import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import argparse
import pdb
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS
from sklearn.covariance import MinCovDet
from sklearn.covariance import GraphicalLasso
from sklearn.decomposition import PCA

# Define the MLP model
class MLP(nn.Module):
	def __init__(self, use_layernorm=False, hidden_size=512):
		super(MLP, self).__init__()
		self.use_layernorm = use_layernorm

		self.fc1 = nn.Linear(784, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 10)

		self.relu = nn.ReLU()
		if self.use_layernorm:
			self.ln1 = nn.LayerNorm(hidden_size)
			self.ln2 = nn.LayerNorm(hidden_size)

	def forward(self, x):
		x = x.view(-1, 784)  # Flatten the input
		x = self.fc1(x)
		if self.use_layernorm:
			x = self.ln1(x)
		x = self.relu(x)
		x = self.fc2(x)
		h = x.detach().clone()
		if self.use_layernorm:
			x = self.ln2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x, h

class LeNet(nn.Module):
	def __init__(self, use_layernorm=False):
		super(LeNet, self).__init__()
		self.use_layernorm = use_layernorm

		# Convolutional layers
		self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
		self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

		# Fully connected layers
		self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16*4*4 input features, 120 output features
		self.fc2 = nn.Linear(120, 84)  # 120 input features, 84 output features
		self.fc3 = nn.Linear(84, 10)  # 84 input features, 10 output features (for 10 classes)

		# Optional LayerNorm
		if self.use_layernorm:
			self.ln1 = nn.LayerNorm([6, 12, 12])
			self.ln2 = nn.LayerNorm([16, 4, 4])
			self.ln3 = nn.LayerNorm(120)
			self.ln4 = nn.LayerNorm(84)

	def forward(self, x):
		# Convolutional layers with ReLU and pooling
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 2x2 max pooling
		if self.use_layernorm:
			x = self.ln1(x)
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # 2x2 max pooling
		if self.use_layernorm:
			x = self.ln2(x)

		# Flatten for fully connected layers
		x = x.view(-1, 16 * 4 * 4)

		# Fully connected layers with ReLU
		x = F.relu(self.fc1(x))
		if self.use_layernorm:
			x = self.ln3(x)
		x = self.fc2(x)
		h = x.detach().clone()
		x = F.relu(x)
		if self.use_layernorm:
			x = self.ln4(x)
		x = self.fc3(x)

		return x, h

# Function to train the model
def train(model, train_loader, optimizer, criterion, device):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output, _ = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

# Function to evaluate the model and collect activations
def evaluate(model, data_loader, device, permute_pixels=False):
	model.eval()
	activations = []
	correct = 0
	total = 0

	# Create a random permutation of the pixels
	permutation = torch.randperm(784) if permute_pixels else torch.arange(784)

	with torch.no_grad():
		for data, target in data_loader:
			data = data.view(-1, 784)[:, permutation].view(-1, 1, 28, 28) # Permute pixels
			data, target = data.to(device), target.to(device)
			output, h = model(data)
			activations.append(h.cpu())

			# Calculate accuracy
			_, predicted = torch.max(output.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()

	accuracy = 100 * correct / total
	return torch.cat(activations), accuracy

def estimate_gaussian_volume(activations, k=2):
	"""
	Estimates the log volume of an n-dimensional ellipsoid approximating a Gaussian
	distribution from activation data.

	Args:
		activations: A numpy array of shape (num_samples, dimensionality) representing the activation data.
		k: The number of standard deviations to use for the ellipsoid boundary.

	Returns:
		The estimated volume of the ellipsoid.
	"""
	# # covariance_matrix = np.cov(activations, rowvar=False)  # rowvar=False means each column is a variable
	# cov = OAS()
	cov = LedoitWolf()
	# cov = MinCovDet(support_fraction = 0.95)
	# cov = GraphicalLasso()
	cov.fit(activations.numpy().astype(np.double)) # more precision
	covariance_matrix = cov.covariance_
	if False:
		# slogdet returns the sign and the log of the determinant
		sign, logdet = np.linalg.slogdet(covariance_matrix)
		n = activations.shape[1]
	if False: # trim the noise eigenvalues
		eigval, eigvec = np.linalg.eigh(covariance_matrix) # checking
		mx = np.mean(np.log(eigval[-14:])) # sorted ascending
		# mx = np.max(np.log(eigval))
		# ghetto version of: https://pmc.ncbi.nlm.nih.gov/articles/PMC3667751/
		logeig = np.log(eigval)
		logeig = np.clip(logeig, mx-10, mx+5)
		n = np.sum(logeig > mx-8)
		logdet = np.sum(logeig * (logeig > mx-8))
		logdet = np.real(logdet) # imaginary component is noise
		cov_cond_no = np.linalg.cond(covariance_matrix)
	if True: 
		# center the activations so SVD is eq. to cov calc
		activations = activations - torch.mean(activations, axis=0)
		U, S, V_transpose = np.linalg.svd( \
			activations.numpy().astype(np.double), full_matrices=False )
		eigval = S**2 # these are sorted descending

		mx = np.mean(np.log(eigval[:16])) # sorted descending
		logeig = np.log(eigval)
		if True:
			logeig = np.clip(logeig, mx-10, mx+5)
			n = np.sum(logeig > mx-8)
			logdet = np.sum(logeig * (logeig > mx-8))
		else:
			n = logeig.shape[0]
			logdet = np.sum(logeig)
		logdet = np.real(logdet) # imaginary component is noise
		joint_entropy = 0.5 * logdet + n/2*(1 + np.log(2*3.1415926))
		# https://math.stackexchange.com/questions/2029707/entropy-of-the-multivariate-gaussian

		covariance_matrix = V_transpose.T @ np.diag( S**2 ) @ V_transpose
		variances = np.diag(covariance_matrix)
		ind_entropy = 0.5 * np.sum(np.log(variances) + np.log(2*3.1315926*2.71828))
		# https://en.wikipedia.org/wiki/Differential_entropy
		# should be OK since the scale is identical.
		mutual_info = ind_entropy - joint_entropy

		cov_cond_no = np.linalg.cond(covariance_matrix)
	if True:
		variances = np.diag(covariance_matrix)
		# sign, logdet_covariance = np.linalg.slogdet(covariance_matrix)
		logdet_variances = np.sum(np.log(variances))
		# mutual info of a multidimensional gaussian is
		# I(X) = (1/2) * log( (2πe)^n * det(Σ) ) - (1/2) * Σᵢ log(2πe * σᵢ²)
		# = 1/2 *( n*log( 2πe ) + log det(Σ) ) - 1/2 Sum_i^n [ log(2πe) + 2*log(σᵢ) ]
		# = 1/2 ( log det(Σ) - Sum_i^n log(σᵢ^2)
		# mutual_info = 0.5 * (logdet_variances - logdet)

		return n, logdet, cov_cond_no

	# volume = (np.pi**(n/2) / math.gamma(n/2 + 1)) * (k**n) * np.sqrt(determinant)
	log_volume = (n/2) * np.log(np.pi) - math.lgamma(n/2 + 1) + n * k + (1/2) * logdet

	return n, log_volume, cov_cond_no


def plot_overlaid_histograms(initial_activations,
									initial_activations_permuted,
									final_activations, final_activations_permuted, title, use_cosine_similarity=False):
	figsize = (12 ,7)
	plt.rcParams['font.size'] = 18
	plt.figure(figsize=figsize)

	# Function to calculate cosine similarities or L2 norms and prepare histogram data
	def plot_hist_data(activations, label, color, linestyle, permute_dim=-1, randn_last_dim=False):
		# center data.
		# activations = activations - torch.mean(activations, 0)
		# approximate (via a gaussian) the volume occupied by the activations.
		if use_cosine_similarity:
			s = torch.std(activations, 0)
			if randn_last_dim:
				# just replace with random normals of the same std
				# (ignore the off-diagonal elements of the covariance)
				a = torch.randn_like(activations) * s[...,:]
				activations = a
			# Permute the last dimension (if requested)
			if permute_dim >= 0:
				if permute_dim == 0:
					for i in range(activations.shape[1]):
						idx = torch.randperm(activations.shape[0])
						activations[:, i] = activations[idx, i]
				if permute_dim == 1:
					for i in range(activations.shape[0]):
						idx = torch.randperm(activations.shape[1])
						activations[i, :] = activations[i, idx]
			# Calculate cosine similarities between all pairs of activations
			norm_activations = F.normalize(activations, p=2, dim=1)  # Normalize to unit vectors
			cosine_similarities = torch.matmul(norm_activations, norm_activations.t())
			# Take upper triangle of the cosine similarity matrix (excluding diagonal)
			mask = torch.triu(torch.ones_like(cosine_similarities), diagonal=1).bool()
			values = cosine_similarities[mask].numpy()
			n, vol, cond_no = estimate_gaussian_volume(activations)
			# print(f'{label}, eig values: {eigval[:10]}')
			print(f'{label}, MI: {vol}, n: {n}, cond_no: {int(cond_no)}')
		else:
			# Calculate L2 norms (vector length)
			values = torch.linalg.norm(activations, dim=1).numpy()

		# need fixed bin edges to make the distributions comparable.
		hist, bin_edges = np.histogram(values, bins=200) # ,range=(-1,1)
		bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

		plt.plot(bin_centers, hist, label=label, color=color, linestyle=linestyle)

		# save the volume to the file
		fid = open('mlp_mnist_table.csv', 'a')
		fid.write(f"{round(vol)}\t")
		fid.close()
		return vol

	fid = open('mlp_mnist_table.csv', 'a')
	fid.write(f"{initial_activations.shape[1]}\t")
	fid.close()

	# Get histogram data for each case, including control with last dimension permutation
	volume_initial = plot_hist_data(initial_activations, \
		"Initial", "blue", "-")
	volume_initial_permute0 = plot_hist_data(initial_activations, \
		"Initial, permuted ctrl 0", "blue", "--", permute_dim=0)
	volume_initial_permute1 = plot_hist_data(initial_activations, \
		"Initial, permuted ctrl 1", "cyan", "--", permute_dim=1)
	# volume_initial_gauss = plot_hist_data(initial_activations, \
	# 	"Initial, gaussian ctrl", "blue", ":", randn_last_dim=True)
	volume_initial_permutepix = plot_hist_data(initial_activations_permuted, \
		"Initial, permuted-pixels", "green", "-.")

	volume_trained = plot_hist_data(final_activations, \
		"Trained", "red", "-")
	volume_trained_permute0 = plot_hist_data(final_activations, \
		"Trained, permuted ctrl0", "red", "--", permute_dim=0)
	volume_trained_permute1 = plot_hist_data(final_activations, \
		"Trained, permuted ctrl1", "magenta", "--", permute_dim=1)
	# volume_trained_gauss = plot_hist_data(final_activations, \
	# 	"Trained, gaussian ctrl", "red", ":", randn_last_dim=True)
	volume_trained_permutepix = plot_hist_data(final_activations_permuted, \
		"Trained, permuted-pixels", "orange", "-.")

	# # print out the volumes.
	# print("Initial, Δ Volume from permutation:", \
	# 	(volume_initial_permute - volume_initial) / math.log(2), "bits")
	# print("Initial, Δ Volume from gauss approx:", \
	# 	(volume_initial_gauss - volume_initial) / math.log(2), "bits")
	# print("Initial, Δ Volume from pixel permute:", \
	# 	(volume_initial_permutepix - volume_initial) / math.log(2), "bits")
 #
	# print("Trained, Δ Volume from permutation:", \
	# 	(volume_trained_permute - volume_trained) / math.log(2), "bits")
	# print("Trained, Δ Volume from gauss approx:", \
	# 	(volume_trained_gauss - volume_trained) / math.log(2), "bits")
	# print("Trained, Δ Volume from pixel permute:", \
	# 	(volume_trained_permutepix - volume_trained) / math.log(2), "bits")
	# if the permutation results in an increase in volume,
	# we can use this to improve the estimate of the volume change from training
	# (the trained network is smaller volume than the cov. matrix estimates)
	# print("Naive Δ Volume from training:", \
	# 	(volume_trained - volume_initial) / math.log(2), "bits")
	# print("Corrected Δ Volume from training:", \
	# 	(volume_trained - volume_trained_permute - volume_initial) / math.log(2), "bits")
	print("Δ Volumes:", \
	 	(volume_initial - volume_initial_permute0), \
		(volume_trained - volume_trained_permute0),)

	fid = open('mlp_mnist_table.csv', 'a')
	fid.write(f"\n")
	fid.close()

	plt.title(title)
	if use_cosine_similarity:
		plt.xlabel("Cosine Similarity of Activations")
	else:
		plt.xlabel("L2 Norm of Activation")
	plt.ylabel("Frequency")
	plt.legend()
	# plt.show()


# Main function
def main():
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description="MLP MNIST Example")
	parser.add_argument("--layer_norm", action="store_true", help="Use LayerNorm")
	parser.add_argument("--reset", action="store_true", help="reset the table")
	parser.add_argument("--lenet", action="store_true", help="Use LeNet5 instead of MLP")
	parser.add_argument('--hidden', type=int, default=512, help='hidden size')
	args = parser.parse_args()

	# Hyperparameters
	batch_size = 64
	epochs = 10
	learning_rate = 1e-3

	# Device configuration
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load MNIST dataset
	train_dataset = datasets.MNIST(
		root="./data", train=True, transform=transforms.ToTensor(), download=True
	)
	test_dataset = datasets.MNIST(
		root="./data", train=False, transform=transforms.ToTensor()
	)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	combined_dataset = ConcatDataset([train_dataset, test_dataset])
	combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

	# Initialize the model
	if args.lenet:
		model = LeNet(use_layernorm=args.layer_norm).to(device)
	else:
		model = MLP(use_layernorm=args.layer_norm, hidden_size=args.hidden).to(device)

	# Optimizer and loss function
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
	criterion = nn.CrossEntropyLoss()

	# Collect activations before training
	initial_activations, initial_accuracy = evaluate(model, test_loader, device)
	initial_activations_permuted, _ = evaluate(model, test_loader, device, permute_pixels=True)
	print(f"Initial Test Accuracy: {initial_accuracy:.2f}%")

	if args.reset:
		fid = open('mlp_mnist_table.csv', 'w')
		fid.write("Hidden\t")
		fid.write("Initial\t")
		fid.write("Init ctrl 0\t")
		fid.write("Init ctrl 1\t")
		# fid.write("Init gauss ctrl\t")
		fid.write("Init permuted pixels\t")
		fid.write("Trained\t")
		fid.write("Train ctrl 0\t")
		fid.write("Train ctrl 1\t")
		# fid.write("Train gauss ctrl\t")
		fid.write("Train permuted pixels\n")
		fid.close()

	# Train the model
	for epoch in range(epochs):
		train(model, train_loader, optimizer, criterion, device)
		print(f"Epoch {epoch+1}/{epochs} completed.")

	# Collect activations after training
	final_activations, final_accuracy = evaluate(model, test_loader, device)
	final_activations_permuted, _ = evaluate(model, test_loader, device, permute_pixels=True)
	print(f"Final Test Accuracy: {final_accuracy:.2f}%")
	plot_overlaid_histograms(
		initial_activations, initial_activations_permuted,
		final_activations, final_activations_permuted,
		"Activations (Layer 2)", use_cosine_similarity=True)

if __name__ == "__main__":
	main()
