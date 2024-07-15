import matplotlib.pyplot as plt
import pandas as pd

# Define the set of heads and layers configurations
heads_layers_set = [
    (1, 1),
    (2, 1),
    (2, 2),
    (4, 4)
    # (8, 8)
]

# Create a figure and axes for subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 15))
axes = axes.flatten()

# Function to create a scatter plot for a given heads and layers configuration
def create_plot(ax, layers, heads):
	# Read the data from the file and store it in a DataFrame
	data = pd.read_csv(f"vallog3_l{layers}_h{heads}.txt", sep="\t", header=None, names=["batch_size", "validation"])

	# Group by batch size and calculate median
	grouped_data = data.groupby("batch_size")["validation"].median().reset_index()

	# Create the scatter plot with a logarithmic y-axis
	ax.scatter(data["batch_size"], data["validation"], c='blue', alpha=0.5, edgecolors='w', s=100, label='Individual runs')

	# Plot median for each batch size
	ax.scatter(grouped_data["batch_size"], grouped_data["validation"], c='black', s=100, label='Median', edgecolors='w')

	ax.set_title(f'{layers} layers, {heads} heads, 15k iters')
	ax.set_xlabel('Number of samples')
	ax.set_ylabel('Validation')
	ax.set_ylim(10**-5, 10**2)
	ax.set_yscale('log')
	ax.grid(True)
	ax.legend()

# Create a plot for each heads and layers configuration
for i, (heads, layers) in enumerate(heads_layers_set):
    create_plot(axes[i], layers, heads)

# Hide any unused subplots
for j in range(len(heads_layers_set), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
