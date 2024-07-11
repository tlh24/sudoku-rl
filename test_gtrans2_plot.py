import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the file and store it in a DataFrame
data = pd.read_csv("vallog.txt", sep="\t", header=None, names=["batch_size", "validation"])

# Group by batch size and calculate mean
# grouped_data = data.groupby("batch_size")["validation"].mean().reset_index()
grouped_data = data.groupby("batch_size")["validation"].median().reset_index()

# Create the scatter plot with a logarithmic y-axis
plt.figure(figsize=(10, 6))
plt.scatter(data["batch_size"], data["validation"], c='blue', alpha=0.5, edgecolors='w', s=100, label='Individual runs')

# Plot mean for each batch size
plt.scatter(grouped_data["batch_size"], grouped_data["validation"], c='black', s=100, label='Median', edgecolors='w')

plt.title('Batch Size vs. Validation: 1 layer, 2 heads, 15k iters')
plt.xlabel('Batch Size')
plt.ylabel('Validation')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()
