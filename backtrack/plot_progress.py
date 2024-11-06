import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file (assuming it's named 'data.csv')
data = pd.read_csv('progress.txt', sep='\t', header=None)

# Extract the second column (index 1)
second_column = data[1]

# Plot the second column
plt.plot(second_column)
plt.xlabel('Index')
plt.ylabel('Values (Second Column)')
plt.title('Plot of the Second Column')
plt.show()
