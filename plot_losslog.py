import numpy as np
import csv
import pdb
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import argparse
import sys
import utils
import glob
# import sklearn

plot_rows = 1
plot_cols = 1
figsize = (12 ,7)
plt.ion()
plt.rcParams['font.size'] = 18
plt.rcParams['figure.dpi'] = 72
fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False

current_directory = os.getcwd()
base_dir = os.path.basename(current_directory)
fig.canvas.manager.set_window_title(f'plot_losslog {base_dir}')

git_commit_hash = utils.getGitCommitHash()

parser = argparse.ArgumentParser(description="Plot loss logs from files in a directory that match a pattern.")
parser.add_argument('-d', '--directory', type=str, default=".", help='Directory to search for log files.')
parser.add_argument('-p', '--pattern', type=str, default="losslog*.txt", help='Pattern to match for log files (e.g., "losslog*.txt").')
cmd_args = parser.parse_args()

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

# Define colors for plotting, use default if not enough colors are provided
colors = ['b', 'r', 'k', 'g', 'm', 'c']  # Extendable list of colors
replicate_colors = 3
color_repeat = [color for color in colors for _ in range(replicate_colors)]

# make a moving-average kernel
window_size = 100
kernel = np.ones(window_size) / window_size

cont = True
while cont:
	ax.cla()
	# Find all files in the directory that match the specified pattern
	search_path = os.path.join(cmd_args.directory, cmd_args.pattern)
	file_names = sorted(glob.glob(search_path))
	# file_names = sorted(file_names, key=os.path.getmtime)

	if not file_names:
		print(f"No files found in '{cmd_args.directory}' matching '{cmd_args.pattern}'")
		time.sleep(2)
		continue

	# Create a color cycle that is long enough for the number of files found
	color_cycle = color_repeat * (len(file_names) // len(color_repeat) + 1)
	# Loop through each file and plot
	for i, fname in enumerate(file_names):
		try:
			with open(fname, 'r') as x:
				data = list(csv.reader(x, delimiter="\t"))
			data = np.array(data)
			data = data.astype(float)

			# Plot the data in log scale for the second column
			if len(data.shape) > 1 and data.shape[0] > 1:
				ax.plot(data[:, 0], np.log(data[:, 1]), color_cycle[i], alpha=0.15, label=f"{os.path.basename(fname)} ({color_cycle[i]})")
				smoothed = np.convolve(data[:, 1], kernel, mode='same')
				if smoothed.shape[0] == data.shape[0]:
					ax.plot(data[:, 0], np.log(smoothed), color_cycle[i], alpha=0.45)
					

		except FileNotFoundError:
			print(f"File {fname} not found. Skipping...")
			continue

	# Add labels, title, and legend
	ax.set(xlabel='iteration / batch #', ylabel='log loss')
	ax.set_title(f'Log Loss Comparison {git_commit_hash}',fontsize=18)
	ax.legend(fontsize=16)

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
