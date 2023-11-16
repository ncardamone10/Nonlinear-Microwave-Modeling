import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# File path to the new data
file_path = "./Nonlinear-Microwave-Modeling/Assignment/Data From ADS/Large Signal S Params/testing13.txt"

# Function to check if a row contains valid data
def is_valid_row(row):
    try:
        float(row[0])
        float(row[1])
        float(row[2])
        return True
    except ValueError:
        return False

# Reading the file and filtering out invalid lines
with open(file_path, 'r') as file:
    lines = file.readlines()
    valid_lines = [line for line in lines if is_valid_row(line.split('\t'))]

# Creating a DataFrame from the valid lines
df = pd.DataFrame([line.split('\t') for line in valid_lines], columns=['Frequency', 'Power', 'S21_dB'])
df = df.astype({'Frequency': 'float', 'Power': 'float', 'S21_dB': 'float'})

# Creating the 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Converting data for plotting
X = df['Frequency']
Y = df['Power']
Z = df['S21_dB']

# Creating a grid for plotting
X_unique = np.sort(df['Frequency'].unique())
Y_unique = np.sort(df['Power'].unique())
X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
Z_grid = np.array([df.loc[(df['Frequency'] == x) & (df['Power'] == y), 'S21_dB'].iloc[0] 
                   for x, y in zip(np.ravel(X_grid), np.ravel(Y_grid))]).reshape(X_grid.shape)

# Plotting the surface
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='jet', edgecolor='none')

# Adding labels and title
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB)')
ax.set_zlabel('S21 (dB)')
ax.set_title('3D Surface Plot of S21 Parameter')

# Adding a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()
