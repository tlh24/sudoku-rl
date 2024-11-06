import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file (assuming it's named 'data.csv')
data = pd.read_csv('progress.txt', sep='\t', header=None)

# Extract the second column (index 1)
second_column = data[1]

moving_average = second_column.rolling(window=100).mean()

# Plot the second column
plt.plot(second_column, 'bo', alpha=0.1)
plt.plot(moving_average, 'k')
plt.xlabel('Index')
plt.ylabel('Number of cells filled')
plt.title('Plot of cells filled vs time')
plt.show()
