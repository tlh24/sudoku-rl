import numpy as np
import csv
import pdb
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import argparse
import sys
# import sklearn

plot_rows = 1
plot_cols = 1
figsize = (7 , 4)
plt.ion()
fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False

current_directory = os.getcwd()
base_dir = os.path.basename(current_directory)
fig.canvas.manager.set_window_title(f'plot_losslog {base_dir}')

# parser = argparse.ArgumentParser(description="Plot a txt file of losses")
# parser.add_argument('-f', type=str, default="./losslog.txt", help='which file to read')
# parser.add_argument('-f2', type=str, default="", help='comparison file')
# cmd_args = parser.parse_args()

def slidingWindowR2(x, y, window_size, stride):
	n = len(x)
	r2_values = []

	for i in range(0, n - window_size + 1, stride):
		x_window = x[i : i + window_size]
		y_window = y[i : i + window_size]
		r2 = np.corrcoef(x_window, y_window)
		r2_values.append(np.clip(r2[0,1], -1, 1))

	return r2_values


def isFileEmpty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) == 0

if "DISPLAY" not in os.environ:
	print("No X11 server detected, switching to non-interactive Agg backend")
	matplotlib.use('Agg')  # Use non-interactive backend

# Check if there are at least two arguments (besides the script name)
if len(sys.argv) < 2:
    file_names = ["losslog.txt"]
else:
	file_names = sys.argv[1:]  # Skip the script name

# Define colors for plotting, use default if not enough colors are provided
colors = ['b', 'k', 'r', 'g', 'm', 'c']  # Extendable list of colors
color_cycle = colors * (len(file_names) // len(colors) + 1)  # Repeat colors if N > len(colors)
cont = True
while cont:
	ax.cla()
	# Loop through each file and plot
	for i, fname in enumerate(file_names):
		try:
			with open(fname, 'r') as x:
					data = list(csv.reader(x, delimiter="\t"))
			data = np.array(data)
			data = data.astype(float)

			# Plot the data in log scale for the second column
			if len(data.shape) > 1 and data.shape[0] > 1:
					ax.plot(data[:, 0], np.log(data[:, 1]), color_cycle[i], alpha=0.5, label=f"{fname} ({color_cycle[i]})")

		except FileNotFoundError:
			print(f"File {fname} not found. Skipping...")
			continue

	# Add labels, title, and legend
	ax.set(xlabel='iteration / batch #', ylabel='log loss')
	ax.set_title('Log Loss Comparison')
	ax.legend()

	fig.tight_layout()
	if matplotlib.get_backend() == 'Agg':
		# Save plot to PNG file when in non-interactive mode
		output_file = 'plot_losslog.png'
		fig.savefig(output_file)
		print(f"Plot saved to {output_file}")
		cont = False
	else:
		fig.canvas.draw()
		fig.canvas.flush_events()
		time.sleep(2)
		print("tock")
