import numpy as np
import csv
import pdb
import matplotlib.pyplot as plt
import time
# import sklearn

# remove menubar buttons
plt.rcParams['toolbar'] = 'None'

plot_rows = 2
plot_cols = 2
figsize = (12, 6)
plt.ion()
fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False

def slidingWindowR2(x, y, window_size, stride):
	n = len(x)
	r2_values = []

	for i in range(0, n - window_size + 1, stride):
		x_window = x[i : i + window_size]
		y_window = y[i : i + window_size]
		r2 = np.corrcoef(x_window, y_window)
		r2_values.append(np.clip(r2[0,1], -1, 1))

	return r2_values


while True: 
	with open("losslog.txt", 'r') as x:
		data = list(csv.reader(x, delimiter="\t"))
	data = np.array(data)
	data = data.astype(float)
	
	if len(data.shape) > 1 and data.shape[0] > 1: 
		ax[0,0].cla()
		ax[0,0].plot(data[:,0], np.log(data[:, 1]), 'b')
		ax[0,0].set(xlabel='iteration')
		ax[0,0].set_title('log loss')
		
		with open("rewardlog.txt", 'r') as x:
			data = list(csv.reader(x, delimiter="\t"))
		data = np.array(data)
		data = data.astype(float)

		ax[0,1].cla()
		r2 = slidingWindowR2(data[:,0], data[:,1], 100, 10)
		ax[0,1].plot(r2)
		ax[0,1].set(xlabel='time')
		ax[0,1].set_title('r^2 of actual vs predicted')
		ax[0,1].tick_params(bottom=True, top=True, left=True, right=True)
		ax[0,1].set_ylim(-0.1, 1.1)
		
		ax[1,1].cla()
		ax[1,1].scatter(data[:,0], data[:, 1], c=range(data.shape[0]), cmap='viridis', s=100)
		ax[1,1].set(xlabel='actual reward')
		ax[1,1].set(ylabel='predicted reward')
		ax[1,1].set_title('reward')
		
		try: 
			ax[1,0].cla()
			y = np.load('prio.npy')
			ax[1,0].plot(y[1,:], 'b')
			ax[1,0].plot(y[2,:], 'r')
			ax[1,0].plot(y[0,:]/100, 'k')
			ax[1,0].set_title('sorted priority replay loss')
		except:
			print('prio.npy not loaded')

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()
	time.sleep(0.5)
	print("tock")
	#plt.show()
