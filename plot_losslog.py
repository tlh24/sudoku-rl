import numpy as np
import csv
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import time
import os
import re
import sys

csv.field_size_limit(sys.maxsize)

plt.ion()
plt.rcParams['font.size'] = 18
plt.rcParams['figure.dpi'] = 72

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.tick_params(axis='y', left=True, right=True, labelleft=True, labelright=True)

if "DISPLAY" in os.environ and matplotlib.get_backend() != 'Agg':
	fig.subplots_adjust(left=0.22)
	ax_check        = fig.add_axes([0.02, 0.35, 0.18, 0.55])
	ax_check.set_axis_off()
	ax_check_metric = fig.add_axes([0.02, 0.05, 0.18, 0.30])
	ax_check_metric.set_axis_off()
else:
	ax_check = None
	ax_check_metric = None

current_directory = os.getcwd()
fig.canvas.manager.set_window_title(f'plot_losslog {os.path.basename(current_directory)}')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default=".")
parser.add_argument('-p', '--pattern',   type=str, default="losslog*.txt")
parser.add_argument('--repl',            type=int, default=1)
cmd_args = parser.parse_args()

visibility_state  = {}  # label -> bool
metric_visibility = {'loss': True, 'top1': True, 'val_loss': True, 'val_top1': True}

METRIC_LABELS = ['Train Loss', 'Top1 Error', 'Val Loss', 'Val Top1 Error']
METRIC_KEYS   = ['loss',       'top1',       'val_loss', 'val_top1']

known_labels = []
all_lines    = []   # (line, label, metric_key)
check_widget = None
check_metric_widget = None

def get_base_label(fname):
	base  = os.path.basename(fname)
	label = re.sub(r'_r\d+\.txt$', '', base).replace('.txt', '')
	for lit in [p for p in re.split(r'[^a-zA-Z0-9]+', cmd_args.pattern) if p]:
		if label.startswith(lit): label = label[len(lit):]
		if label.endswith(lit):   label = label[:-len(lit)]
	label = label.strip('_- .')
	return label or base.replace('.txt', '')

def update_all_visibility():
	for line, label, metric in all_lines:
		line.set_visible(visibility_state.get(label, True) and metric_visibility.get(metric, True))
	fig.canvas.draw_idle()

def toggle_visibility(label):
	visibility_state[label] = not visibility_state[label]
	update_all_visibility()

def toggle_metric(display_label):
	key = METRIC_KEYS[METRIC_LABELS.index(display_label)]
	metric_visibility[key] = not metric_visibility[key]
	update_all_visibility()

if "DISPLAY" not in os.environ:
	print("No X11 server detected, switching to non-interactive Agg backend")
	matplotlib.use('Agg')

colors = [
	'#d62728', '#1f77b4', '#ff7f0e', '#bcbd22', '#2ca02c',
	'#17becf', '#9467bd', '#e377c2', '#8c564b', '#7f7f7f'
]
replicate_colors = cmd_args.repl
color_repeat = [c for c in colors for _ in range(replicate_colors)]

window_size = 512
kernel = 1 - np.cos(np.linspace(0, 2*3.1415926, window_size))
kernel /= np.sum(kernel)

try:
	regex = re.compile(cmd_args.pattern)
except re.error as e:
	print(f"Invalid regex: {e}"); exit(1)

cont = True
while cont:
	try:
		all_items = os.listdir(cmd_args.directory)
	except FileNotFoundError:
		print(f"Directory not found: {cmd_args.directory}"); all_items = []

	file_names = sorted(
		os.path.join(cmd_args.directory, f)
		for f in all_items
		if os.path.isfile(os.path.join(cmd_args.directory, f)) and regex.search(f)
	)

	if not file_names:
		print(f"No files found in '{cmd_args.directory}' matching '{cmd_args.pattern}'")
		if matplotlib.get_backend() != 'Agg':
			for _ in range(20): fig.canvas.flush_events(); time.sleep(0.1)
		else:
			time.sleep(2)
		continue

	current_labels = []
	for fname in file_names:
		label = get_base_label(fname)
		if label not in current_labels:
			current_labels.append(label)
			visibility_state.setdefault(label, True)

	# Rebuild source-file checkboxes when label set changes
	if ax_check is not None and set(current_labels) != set(known_labels):
		known_labels = current_labels.copy()
		ax_check.clear(); ax_check.set_axis_off()
		with plt.rc_context({'font.size': 16}):
			check_widget = CheckButtons(ax_check, known_labels, [visibility_state[l] for l in known_labels])
		check_widget.on_clicked(toggle_visibility)
		for t in check_widget.labels: t.set_fontsize(16)

	# Create metric checkboxes once
	if ax_check_metric is not None and check_metric_widget is None:
		with plt.rc_context({'font.size': 16}):
			check_metric_widget = CheckButtons(ax_check_metric, METRIC_LABELS,
				[metric_visibility[k] for k in METRIC_KEYS])
		check_metric_widget.on_clicked(toggle_metric)
		for t in check_metric_widget.labels: t.set_fontsize(16)

	ax.cla()
	all_lines.clear()

	color_cycle = color_repeat * (len(file_names) // len(color_repeat) + 1)

	for i, fname in enumerate(file_names):
		label  = get_base_label(fname)
		lv     = visibility_state.get(label, True)
		is_r1  = fname.endswith('_r1.txt') or not any(fname.endswith(f'_r{r}.txt') for r in range(1, 10))
		c      = color_cycle[i]

		try:
			with open(fname) as f:
				data = np.array(list(csv.reader(f, delimiter="\t")), dtype=float)

			if data.ndim < 2 or data.shape[0] < 2:
				continue

			xs = data[:, 0]

			# Col 1: train loss — faint raw + smoothed solid
			ml = metric_visibility['loss']
			line_raw, = ax.plot(xs, np.log(data[:, 1]), c, alpha=0.05, visible=lv and ml)
			all_lines.append((line_raw, label, 'loss'))
			smoothed = np.convolve(data[:, 1], kernel, mode='same')[:len(xs)]
			kw = dict(color=c, alpha=1.0, linewidth=2, visible=lv and ml)
			if is_r1:
				line_sm, = ax.plot(xs, np.log(smoothed), label=label, **kw)
			else:
				line_sm, = ax.plot(xs, np.log(smoothed), **kw)
			all_lines.append((line_sm, label, 'loss'))

			# Col 2: top1 error — faint raw + smoothed solid (clipped to avoid -inf at 0)
			mv = metric_visibility['top1']
			line_top1_raw, = ax.plot(xs, np.log(data[:, 2]).clip(-12), c, alpha=0.05, visible=lv and mv)
			all_lines.append((line_top1_raw, label, 'top1'))
			smoothed_top1 = np.convolve(data[:, 2], kernel, mode='same')[:len(xs)]
			line_top1_sm, = ax.plot(xs, np.log(smoothed_top1).clip(-12), c, alpha=1.0, linewidth=2, visible=lv and mv)
			all_lines.append((line_top1_sm, label, 'top1'))

			# Col 3: val loss — dashed, no smoothing
			mv = metric_visibility['val_loss']
			line_vl, = ax.plot(xs, np.log(data[:, 3]), c, alpha=0.7, linewidth=2, visible=lv and mv)
			all_lines.append((line_vl, label, 'val_loss'))

			# Col 4: val top1 error — dashed, no smoothing
			mv = metric_visibility['val_top1']
			line_vt, = ax.plot(xs, np.log(data[:, 4]).clip(-12), c, alpha=0.7, linewidth=2, visible=lv and mv)
			all_lines.append((line_vt, label, 'val_top1'))

		except FileNotFoundError:
			print(f"File {fname} not found. Skipping...")
		except (ValueError, IndexError) as e:
			print(f"Skipping {fname} (likely mid-write): {e}")

	ax.set(xlabel='iteration / batch #', ylabel='log loss')
	ax.set_title('Log Loss Comparison', fontsize=18)
	leg = ax.legend(fontsize=16)
	for line in leg.get_lines():
		line.set_linewidth(8.0)
		line.set_alpha(1.0)
		line.set_visible(True)

	if ax_check is not None:
		fig.tight_layout(rect=[0.18, 0, 1, 1])
	else:
		fig.tight_layout()

	if matplotlib.get_backend() == 'Agg':
		fig.savefig('plot_losslog.png')
		print("Plot saved to plot_losslog.png")
		cont = False
	else:
		fig.canvas.draw()
		for _ in range(20): fig.canvas.flush_events(); time.sleep(0.1)
		print("tock")
