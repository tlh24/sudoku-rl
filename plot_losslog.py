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

# Terminate the script when the plot window is closed.
window_open = True
def _on_close(event):
	global window_open
	window_open = False
fig.canvas.mpl_connect('close_event', _on_close)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default=".")
parser.add_argument('-p', '--pattern', type=str, default="losslog*.txt")
parser.add_argument('--repl', type=int, default=1)
parser.add_argument('--stats', action='store_true',
						default=False,
						help='plot mean ± 1 std across replicates instead of individual lines')
parser.add_argument('--smooth', type=int, default=512,
						help='convolutional smoothing window width (2 or less turns smoothing off)')
parser.add_argument('--compare', type=str, default=None,
						help='space-separated config tokens (e.g. "hg_ g_"): files identical '
						     'except for these tokens are paired in one color, first token '
						     'solid, the rest dashed')
cmd_args = parser.parse_args()

# Tokens that distinguish otherwise-identical configs (binary/N-ary variate).
compare_tokens = cmd_args.compare.split() if cmd_args.compare else []
# Linestyle per compare token: first solid, then dashed, dotted, dash-dot.
COMPARE_STYLES = ['-', '--', ':', '-.']

visibility_state  = {}  # label -> bool
metric_visibility = {'loss': True, 'top1': False}

METRIC_LABELS = ['Loss', 'Top1 Error']
METRIC_KEYS   = ['loss', 'top1']

known_labels = []
all_lines    = []   # (line, label, metric_key)
check_widget = None
check_metric_widget = None
check_groups = {}   # checkbox display key -> [labels it controls]

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

def toggle_visibility(display_key):
	# In compare mode a checkbox controls a whole group (e.g. both hg_ and g_);
	# otherwise it controls a single label.
	labels = check_groups.get(display_key, [display_key])
	new_state = not visibility_state.get(labels[0], True)
	for l in labels:
		visibility_state[l] = new_state
	update_all_visibility()

def toggle_metric(display_label):
	key = METRIC_KEYS[METRIC_LABELS.index(display_label)]
	metric_visibility[key] = not metric_visibility[key]
	update_all_visibility()

if "DISPLAY" not in os.environ:
	print("No X11 server detected, switching to non-interactive Agg backend")
	matplotlib.use('Agg')

# --- Discrete, baked-in color choices (kept in case we want to revert) ---
# colors = [
# 	'#d62728', '#1f77b4', '#ff7f0e', '#bcbd22', '#2ca02c',
# 	'#17becf', '#9467bd', '#e377c2', '#8c564b', '#7f7f7f'
# ]
# replicate_colors = cmd_args.repl
# color_repeat = [c for c in colors for _ in range(replicate_colors)]

def hamming(a, b):
	"""Hamming distance between two strings, right-padded to equal length."""
	n = max(len(a), len(b))
	a = a.ljust(n); b = b.ljust(n)
	return sum(c1 != c2 for c1, c2 in zip(a, b))

def order_by_hamming(labels):
	"""Greedy nearest-neighbour (topological) ordering by Hamming distance,
	so that similar labels sit adjacent in the legend and colormap."""
	remaining = sorted(labels)          # deterministic start point
	if not remaining:
		return []
	order = [remaining.pop(0)]
	while remaining:
		last = order[-1]
		nxt = min(remaining, key=lambda l: (hamming(last, l), l))
		order.append(nxt); remaining.remove(nxt)
	return order

# Continuous colormap for the lines. JET (kept below in case we want it back)
# reads poorly on white: its yellow/cyan bands nearly disappear. A truncated,
# perceptually-uniform map keeps lines legible and colorblind-friendly.
# Alternatives to try: 'plasma', 'turbo', 'cividis'.
LINE_CMAP     = 'turbo'
CMAP_RANGE    = (0.0, 1.0)   # drop the light-yellow top end so lines stay visible on white
# LINE_CMAP   = 'jet'         # original choice
# CMAP_RANGE  = (0.0, 1.0)

def split_compare(label):
	"""Return (base_key, token_index) for the compare token found in `label`,
	or (label, None) if none match. Longer tokens are tried first so that,
	e.g., 'hg_' wins over the 'g_' it contains as a substring."""
	for tok in sorted(compare_tokens, key=len, reverse=True):
		if tok in label:
			return label.replace(tok, '', 1), compare_tokens.index(tok)
	return label, None

def make_label_colors(ordered_labels):
	"""Continuous colors spread across the (Hamming-ordered) label set."""
	n    = len(ordered_labels)
	cmap = plt.get_cmap(LINE_CMAP)
	lo, hi = CMAP_RANGE
	return {l: cmap(lo + (hi - lo) * (i / (n - 1) if n > 1 else 0.5))
	        for i, l in enumerate(ordered_labels)}

window_size = cmd_args.smooth
if window_size <= 2:
	kernel = np.array([1.0])  # identity: convolution leaves data unchanged
else:
	kernel = 1 - np.cos(np.linspace(0, 2*3.1415926, window_size))
	kernel /= np.sum(kernel)

try:
	regex = re.compile(cmd_args.pattern)
except re.error as e:
	print(f"Invalid regex: {e}"); exit(1)

cont = True
while cont and window_open:
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
			for _ in range(20):
				if not window_open: break
				fig.canvas.flush_events(); time.sleep(0.1)
		else:
			time.sleep(2)
		continue

	current_labels = []
	for fname in file_names:
		label = get_base_label(fname)
		if label not in current_labels:
			current_labels.append(label)
			visibility_state.setdefault(label, True)

	# Order labels by Hamming distance (topological sort) and assign
	# continuous colors in that order.
	current_labels = order_by_hamming(current_labels)

	if compare_tokens:
		# Pair labels that are identical except for the compare token: each
		# pair shares a color (from the Hamming-ordered base keys) and is
		# distinguished by linestyle. Labels matching no token fall back to
		# their own color + solid line (partner may not exist yet — realtime).
		base_of = {l: split_compare(l) for l in current_labels}   # l -> (base, idx)
		bases   = []
		for l in current_labels:
			b = base_of[l][0]
			if b not in bases:
				bases.append(b)
		bases       = order_by_hamming(bases)
		base_color  = make_label_colors(bases)
		label_color = {l: base_color[base_of[l][0]] for l in current_labels}
		label_style = {l: (COMPARE_STYLES[base_of[l][1] % len(COMPARE_STYLES)]
		                   if base_of[l][1] is not None else '-')
		               for l in current_labels}
		# Reorder so paired variants sit together, following base Hamming order.
		base_rank = {b: i for i, b in enumerate(bases)}
		current_labels = sorted(
			current_labels,
			key=lambda l: (base_rank[base_of[l][0]],
			               base_of[l][1] if base_of[l][1] is not None else -1))
	else:
		label_color = make_label_colors(current_labels)
		label_style = {l: '-' for l in current_labels}

	# Rebuild source-file checkboxes when label set changes
	if ax_check is not None and set(current_labels) != set(known_labels):
		known_labels = current_labels.copy()
		# In compare mode, tie all variants of a base key to one checkbox;
		# otherwise one checkbox per label. current_labels is already grouped
		# so variants stay adjacent and keys follow the Hamming ordering.
		check_groups = {}
		if compare_tokens:
			for l in current_labels:
				check_groups.setdefault(split_compare(l)[0], []).append(l)
		else:
			for l in current_labels:
				check_groups[l] = [l]
		display_keys = list(check_groups.keys())
		ax_check.clear(); ax_check.set_axis_off()
		with plt.rc_context({'font.size': 16}):
			check_widget = CheckButtons(ax_check, display_keys,
				[visibility_state.get(check_groups[k][0], True) for k in display_keys])
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

	if cmd_args.stats:
		from collections import defaultdict
		label_to_files = defaultdict(list)
		for fname in file_names:
			label_to_files[get_base_label(fname)].append(fname)

		for i, label in enumerate(current_labels):
			fnames = label_to_files[label]
			lv = visibility_state.get(label, True)
			c  = label_color[label]
			ls = label_style[label]

			arrays = []
			for fname in fnames:
				try:
					with open(fname) as f:
						rows = list(csv.reader(f, delimiter='\t'))
					if rows:
						ncols = max(len(r) for r in rows)
						rows = [r for r in rows if len(r) == ncols]
					data = np.array(rows, dtype=float)
					if data.ndim >= 2 and data.shape[0] >= 2:
						arrays.append(data)
				except FileNotFoundError:
					print(f"File {fname} not found. Skipping...")
				except (ValueError, IndexError) as e:
					print(f"Skipping {fname} (likely mid-write): {e}")

			if not arrays:
				continue

			min_len = min(a.shape[0] for a in arrays)
			xs = arrays[0][:min_len, 0]

			def _plot_stat(col, metric_key, transform, first_label, _min_len=min_len):
				mv  = metric_visibility[metric_key]
				vis = lv and mv
				vals = np.stack([transform(a[:_min_len, col]) for a in arrays])
				mean = vals.mean(axis=0)
				std  = vals.std(axis=0)
				line, = ax.plot(xs, mean, color=c, linewidth=2, alpha=1.0, linestyle=ls,
				                label=label if first_label else None, visible=vis)
				all_lines.append((line, label, metric_key))
				fill = ax.fill_between(xs, mean - std, mean + std,
				                       color=c, alpha=0.2, visible=vis)
				all_lines.append((fill, label, metric_key))

			# Only the solid variant of a compare group carries a legend entry.
			_plot_stat(1, 'loss',
			           lambda col, k=min_len: np.log(np.convolve(col, kernel, mode='same')[:k]),
			           first_label=(ls == '-'))
			_plot_stat(2, 'top1',
			           lambda col, k=min_len: np.log(np.convolve(col, kernel, mode='same')[:k]).clip(-12),
			           first_label=False)

	else:
		# Plot in Hamming order so the legend follows the sorted label set.
		label_rank = {l: i for i, l in enumerate(current_labels)}
		file_names_ordered = sorted(
			file_names, key=lambda f: (label_rank.get(get_base_label(f), 0), f))

		for fname in file_names_ordered:
			label  = get_base_label(fname)
			lv     = visibility_state.get(label, True)
			is_r1  = fname.endswith('_r1.txt') or not any(fname.endswith(f'_r{r}.txt') for r in range(1, 10))
			c      = label_color[label]
			ls     = label_style[label]

			try:
				with open(fname) as f:
					rows = list(csv.reader(f, delimiter="\t"))
				if rows:
					ncols = max(len(r) for r in rows)
					rows = [r for r in rows if len(r) == ncols]
				data = np.array(rows, dtype=float)

				if data.ndim < 2 or data.shape[0] < 2:
					continue

				xs = data[:, 0]

				# Col 1: train loss — faint raw + smoothed solid
				ml = metric_visibility['loss']
				line_raw, = ax.plot(xs, np.log(data[:, 1]), color=c, alpha=0.05, visible=lv and ml)
				all_lines.append((line_raw, label, 'loss'))
				smoothed = np.convolve(data[:, 1], kernel, mode='same')[:len(xs)]
				kw = dict(color=c, alpha=1.0, linewidth=2, linestyle=ls, visible=lv and ml)
				# Only the solid variant of a compare group carries a legend entry.
				leg_label = label if (is_r1 and ls == '-') else None
				line_sm, = ax.plot(xs, np.log(smoothed), label=leg_label, **kw)
				all_lines.append((line_sm, label, 'loss'))

				# Col 2: top1 error — faint raw + smoothed solid (clipped to avoid -inf at 0)
				mv = metric_visibility['top1']
				line_top1_raw, = ax.plot(xs, np.log(data[:, 2]).clip(-12), color=c, alpha=0.05, visible=lv and mv)
				all_lines.append((line_top1_raw, label, 'top1'))
				smoothed_top1 = np.convolve(data[:, 2], kernel, mode='same')[:len(xs)]
				line_top1_sm, = ax.plot(xs, np.log(smoothed_top1).clip(-12), color=c, alpha=1.0, linewidth=2, linestyle=ls, visible=lv and mv)
				all_lines.append((line_top1_sm, label, 'top1'))

			except FileNotFoundError:
				print(f"File {fname} not found. Skipping...")
			except (ValueError, IndexError) as e:
				print(f"Skipping {fname} (likely mid-write): {e}")

	ax.set(xlabel='iteration / batch #', ylabel='log loss')
	ax.set_title('Log Loss Comparison', fontsize=18)
	leg = ax.legend(fontsize=16,loc='lower left')
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
		for _ in range(20):
			if not window_open: break
			fig.canvas.flush_events(); time.sleep(0.1)
		if window_open:
			print("tock")
