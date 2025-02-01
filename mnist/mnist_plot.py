import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = ['mnist_zeroinit_4.txt', 'mnist_zeroinit_4.txt']
widths = [[784, 512, 10],[784, 1512, 10]]


# Second Scatter Plot: Accuracy vs Sparsity
fix,axs = plt.subplots(2, 3, figsize=(16, 10))
for j, fname in enumerate(file_path):
	df = pd.read_csv(fname, delimiter='\t')
	df['condition'] = df['zero_init'].astype(str) + '_' + df['optimizer']
	for i in range(3):
		filtered = df[df['condition'] == 'True_adagrad']
		axs[j,i].scatter(filtered[f'sparsity_{i}'], filtered['accuracy'], label=f'sparsity_{i}', color='r')
		filtered = df[df['condition'] == 'False_adagrad']
		axs[j,i].scatter(filtered[f'sparsity_{i}'], filtered['accuracy'], label=f'sparsity_{i}', color='b')

		axs[j,i].set(ylabel ='Accuracy')
		axs[j,i].set(xlabel ='Sparsity')
		axs[j,i].set_title(f'Accuracy vs Sparsity for layer {i} width= {widths[j][i]}')
		axs[j,i].grid(True)
		# axs[j,i].set_ylim([0.6, 0.8])

plt.tight_layout()
plt.savefig('MNIST_accuracy_vs_sparsity.pdf')
plt.show()
