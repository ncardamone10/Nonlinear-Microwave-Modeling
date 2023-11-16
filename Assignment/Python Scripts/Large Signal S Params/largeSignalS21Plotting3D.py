import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the Excel file
file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/Large Signal S Params/largeSignalS21_VDS=2V_VGS=-0.7V.xlsx'
data = pd.read_excel(file_path)

# Convert frequency to GHz
data['RFfreq'] = data['RFfreq'] / 1e9

# Calculate the magnitude of S21 in dB
data['magnitude_S21_dB'] = 20 * np.log10(np.sqrt(data['real(S(2,1))']**2 + data['imag(S(2,1))']**2))

# Create a figure with subplots
fig = plt.figure(figsize=(10, 14))

# Top subplot: Magnitude surface plot
ax1 = fig.add_subplot(211, projection='3d')

# Create a grid for plotting
X, Y = np.meshgrid(np.unique(data['RFfreq']), np.unique(data['RFpower']))
Z_magnitude = data.pivot_table(index='RFpower', columns='RFfreq', values='magnitude_S21_dB').values

# Plotting the magnitude surface with jet colormap
surf1 = ax1.plot_surface(X, Y, Z_magnitude, cmap='jet', edgecolor='none')

# Labels and title for magnitude plot
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('RF Power')
ax1.set_zlabel('Magnitude of S21 (dB)')
ax1.set_title('Magnitude of S21 (dB)')

# Create a color bar for magnitude plot
cbar1 = fig.colorbar(surf1, ax=ax1, pad=0.1, aspect=30, format='%d dB')

# Bottom subplot: Phase surface plot
ax2 = fig.add_subplot(212, projection='3d')

# Calculate the phase of S21 in degrees
data['phase_S21_deg'] = np.degrees(np.arctan2(data['imag(S(2,1))'], data['real(S(2,1))']))

# Create a grid for plotting
Z_phase = data.pivot_table(index='RFpower', columns='RFfreq', values='phase_S21_deg').values

# Plotting the phase surface with jet colormap
surf2 = ax2.plot_surface(X, Y, Z_phase, cmap='jet', edgecolor='none')

# Set limits for the z-axis in the phase plot
#ax2.set_zlim(-180, 180)

# Set tick marks on the z-axis to go from -180 to 180 in 45-degree increments
ax2.set_zticks(np.arange(-180, 181, 45))

# Labels and title for phase plot
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('RF Power')
ax2.set_zlabel('Phase of S21 (degrees)')
ax2.set_title('Phase of S21 (degrees)')

# Create a color bar for phase plot with tick marks
cbar2 = fig.colorbar(surf2, ax=ax2, pad=0.1, aspect=30, format='%d degrees', ticks=np.arange(-180, 181, 45))

# Adjust subplot spacing
plt.subplots_adjust(hspace=0.3)

# Show the plot
plt.show()
