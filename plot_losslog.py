import numpy as np
import csv
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons  # NEW: Import CheckButtons
import time
import os
import sys
import glob
import re

plot_rows = 1
plot_cols = 1
figsize = (12, 8)
plt.ion()
plt.rcParams['font.size'] = 18
plt.rcParams['figure.dpi'] = 72

fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
ax.tick_params(axis='y', left=True, right=True, labelleft=True, labelright=True)

# NEW: Create a dedicated axes area for the checkboxes on the left side
if "DISPLAY" in os.environ and matplotlib.get_backend() != 'Agg':
	fig.subplots_adjust(left=0.25)
	# [left, bottom, width, height]
	ax_check = fig.add_axes([0.02, 0.4, 0.18, 0.4])
	ax_check.set_axis_off()
else:
	ax_check = None

initialized = False

current_directory = os.getcwd()
base_dir = os.path.basename(current_directory)
fig.canvas.manager.set_window_title(f'plot_losslog {base_dir}')

git_commit_hash = ""

parser = argparse.ArgumentParser(description="Plot loss logs from files in a directory that match a pattern.")
parser.add_argument('-d', '--directory', type=str, default=".", help='Directory to search for log files.')
parser.add_argument('-p', '--pattern', type=str, default="losslog*.txt", help='Pattern to match for log files (e.g., "losslog*.txt").')
parser.add_argument('--repl', type=int, default=1, help="how many replicates there are",)
cmd_args = parser.parse_args()

visibility_state = {}
known_labels = []
lines_by_label = {}
check_widget = None

def get_base_label(fname):
	"""Extracts the base name, removing the replicate suffix and trimming pattern literals."""
	base = os.path.basename(fname)
	# Remove replicate suffix and extension
	label = re.sub(r'_r\d+\.txt$', '', base)
	label = label.replace('.txt', '')
	# 1. Extract alphanumeric words from the user's pattern
	# E.g., "losslog*.txt" -> ["losslog", "txt"]
	pattern_literals = [p for p in re.split(r'[^a-zA-Z0-9]+', cmd_args.pattern) if p]
	# 2. Trim these literal words from the start/end of the label
	for literal in pattern_literals:
		if label.startswith(literal):
			label = label[len(literal):]
		if label.endswith(literal):
			label = label[:-len(literal)]
	# 3. Clean up any leftover leading/trailing underscores, hyphens, or dots
	label = label.strip('_- .')
	# 4. Fallback in case trimming accidentally removes everything
	if not label:
		label = base.replace('.txt', '')
	return label

def toggle_visibility(label):
	"""Callback function triggered when a CheckButton is clicked."""
	visibility_state[label] = not visibility_state[label]
	# Update the visibility of the lines instantly without waiting for the next reload
	for line in lines_by_label.get(label, []):
		line.set_visible(visibility_state[label])
	fig.canvas.draw_idle()

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

colors = [
    '#d62728', '#1f77b4', '#ff7f0e', '#bcbd22', '#2ca02c',
    '#17becf', '#9467bd', '#e377c2', '#8c564b', '#7f7f7f'
]
replicate_colors = cmd_args.repl
color_repeat = [color for color in colors for _ in range(replicate_colors)]

window_size = 128
kernel = 1-(np.cos(np.linspace(0, 2*3.1415926, window_size)))
kernel = kernel / np.sum(kernel)

try:
	regex = re.compile(cmd_args.pattern)
except re.error as e:
	print(f"Invalid regex: {e}")
	exit(1)

cont = True
while cont:
	file_names = []
	try:
		all_items = os.listdir(cmd_args.directory)
	except FileNotFoundError:
		print(f"Directory not found: {cmd_args.directory}")
		all_items = []
	matched_files = []
	for item in all_items:
		full_path = os.path.join(cmd_args.directory, item)
		if os.path.isfile(full_path) and regex.search(item):
			matched_files.append(full_path)
	file_names = sorted(matched_files)

	if not file_names:
		print(f"No files found in '{cmd_args.directory}' matching '{cmd_args.pattern}'")
		if matplotlib.get_backend() != 'Agg':
			# NEW: Responsive sleep loop
			for _ in range(20):
				fig.canvas.flush_events()
				time.sleep(0.1)
		else:
			time.sleep(2)
		continue

	current_labels = []
	for fname in file_names:
		label = get_base_label(fname)
		if label not in current_labels:
			current_labels.append(label)
			if label not in visibility_state:
				visibility_state[label] = True # Default to visible

	if ax_check is not None and set(current_labels) != set(known_labels):
		known_labels = current_labels.copy()
		ax_check.clear()
		ax_check.set_axis_off()
		states = [visibility_state[l] for l in known_labels]
		# Store in global variable so it isn't garbage collected
		with plt.rc_context({'font.size': 16}):
			check_widget = CheckButtons(ax_check, known_labels, states)
		check_widget.on_clicked(toggle_visibility)
		for text in check_widget.labels:
			text.set_fontsize(16)
	# ------------------------------------------------------

	ax.cla()
	lines_by_label.clear()

	color_cycle = color_repeat * (len(file_names) // len(color_repeat) + 1)

	for i, fname in enumerate(file_names):
		label = get_base_label(fname)
		is_visible = visibility_state.get(label, True)

		if label not in lines_by_label:
			lines_by_label[label] = []

		try:
			with open(fname, 'r') as x:
				data = list(csv.reader(x, delimiter="\t"))
			data = np.array(data)
			data = data.astype(float)

			if len(data.shape) > 1 and data.shape[0] > 1:
				# Track the raw plotted line
				line_raw, = ax.plot(data[:, 0], np.log(data[:, 1]), color_cycle[i], alpha=0.05, visible=is_visible)
				lines_by_label[label].append(line_raw)

				smoothed = np.convolve(data[:, 1], kernel, mode='same')
				if smoothed.shape[0] == data.shape[0]:
					if fname.endswith('_r1.txt') or not any(fname.endswith(f'_r{r}.txt') for r in range(1, 10)):
						# Track the smoothed line (with label)
						line_smooth, = ax.plot(data[:, 0], np.log(smoothed), color_cycle[i], alpha=1.0, label=label, linewidth=2, visible=is_visible)
					else:
						# Track the smoothed line (without label)
						line_smooth, = ax.plot(data[:, 0], np.log(smoothed), color_cycle[i], alpha=1.0, linewidth=2, visible=is_visible)

					lines_by_label[label].append(line_smooth)

		except FileNotFoundError:
			print(f"File {fname} not found. Skipping...")
			continue

	ax.set(xlabel='iteration / batch #', ylabel='log loss')
	ax.set_title(f'Log Loss Comparison {git_commit_hash}',fontsize=18)
	leg = ax.legend(fontsize=16)

	for line in leg.get_lines():
		line.set_linewidth(8.0)

	# NEW: Restrict tight_layout from overlapping the custom CheckButtons axis
	if ax_check is not None:
		fig.tight_layout(rect=[0.18, 0, 1, 1])
	else:
		fig.tight_layout()

	if matplotlib.get_backend() == 'Agg':
		output_file = 'plot_losslog.png'
		fig.savefig(output_file)
		print(f"Plot saved to {output_file}")
		cont = False
	else:
		fig.canvas.draw()
		# NEW: Non-blocking sleep loop to ensure UI doesn't freeze and clicks register
		for _ in range(20):
			fig.canvas.flush_events()
			time.sleep(0.1)
		print("tock")
