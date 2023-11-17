# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import RectBivariateSpline
# import numpy.ma as ma

# # Read the data from the Excel file
# excel_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV_Curves_Updated.xlsx'  # Update this path to your data file location
# data = pd.read_excel(excel_file_path)

# # Convert and clean data
# data['VGS'] = pd.to_numeric(data['VGS'], errors='coerce')
# data['VDS'] = pd.to_numeric(data['VDS'], errors='coerce')
# data['IDS'] = pd.to_numeric(data['IDS'], errors='coerce')
# data.dropna(inplace=True)

# # Create a pivot table for VGS and VDS
# pivot_table = data.pivot_table(values='IDS', index='VGS', columns='VDS')

# # Convert index/columns to NumPy array for RectBivariateSpline
# vgs_values = pivot_table.index.to_numpy()
# vds_values = pivot_table.columns.to_numpy()

# # Bivariate spline approximation
# spline = RectBivariateSpline(vgs_values, vds_values, pivot_table)

# # Calculate partial derivatives
# vgs_values_2d = vgs_values[:, np.newaxis]
# partial_derivative_vgs = spline.partial_derivative(1, 0)(vgs_values_2d, vds_values)
# partial_derivative_vds = spline.partial_derivative(0, 1)(vgs_values_2d, vds_values)

# gm = partial_derivative_vgs
# # Avoid division by zero in r0 calculation
# r0 = 1 / (partial_derivative_vds + 1e-12)

# # Limit r0 to 10 MΩ
# r0_masked = ma.masked_where((r0 > 1e6), r0)

# Av = gm * r0
# # Convert Av to dB
# Av_dB = 20 * np.log10(np.abs(Av))
# Av_dB_masked = ma.masked_where((Av_dB < 0) | (Av_dB > 60), Av_dB)  # Adjust dB limits as necessary

# # Meshgrid for plotting
# vgs_mesh, vds_mesh = np.meshgrid(vgs_values, vds_values)

# # Transpose Av_dB_masked to match the shape of the meshgrid
# Av_dB_masked_reshaped = Av_dB_masked.T

# # Transpose pivot_table.values to match the meshgrid
# IDS_reshaped = (1000 * pivot_table.values).T  # Converting IDS to mA

# # Create a figure for the 3D plot and 2D scatter plot
# fig = plt.figure(figsize=(18, 10))

# # 3D surface plot
# ax1 = fig.add_subplot(121, projection='3d')
# surf = ax1.plot_surface(vds_mesh, IDS_reshaped, Av_dB_masked_reshaped, cmap='jet')
# ax1.set_xlabel('VDS (V)')
# ax1.set_ylabel('IDS (mA)')
# ax1.set_zlabel('Av (dB)')
# ax1.set_title('3D Surface Plot of VDS vs IDS vs Av in dB')

# # 2D scatter plot
# ax2 = fig.add_subplot(122)
# sc = ax2.scatter(vds_mesh.flatten(), IDS_reshaped.flatten(), c=Av_dB_masked_reshaped.flatten(), cmap='jet')
# ax2.set_xlabel('VDS (V)')
# ax2.set_ylabel('IDS (mA)')
# ax2.set_title('2D Scatter Plot of VDS vs IDS (Color: Av in dB)')

# # Add a color bar for the scatter plot
# fig.colorbar(sc, ax=ax2, shrink=0.5, aspect=5)

# # Show the plot
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
import numpy.ma as ma
from matplotlib.ticker import FuncFormatter

# Function to format ticks with 'dB'
def format_ticks(x, pos):
    return f'{int(x)} dB'

# Read the data from the Excel file
excel_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV_Curves_Updated.xlsx'
data = pd.read_excel(excel_file_path)

# Convert and clean data
data['VGS'] = pd.to_numeric(data['VGS'], errors='coerce')
data['VDS'] = pd.to_numeric(data['VDS'], errors='coerce')
data['IDS'] = pd.to_numeric(data['IDS'], errors='coerce')
data.dropna(inplace=True)

# Create a pivot table for VGS and VDS
pivot_table = data.pivot_table(values='IDS', index='VGS', columns='VDS')

# Convert index/columns to NumPy array for RectBivariateSpline
vgs_values = pivot_table.index.to_numpy()
vds_values = pivot_table.columns.to_numpy()

# Bivariate spline approximation
spline = RectBivariateSpline(vgs_values, vds_values, pivot_table)

# Calculate partial derivatives
vgs_values_2d = vgs_values[:, np.newaxis]
partial_derivative_vgs = spline.partial_derivative(1, 0)(vgs_values_2d, vds_values)
partial_derivative_vds = spline.partial_derivative(0, 1)(vgs_values_2d, vds_values)

gm = partial_derivative_vgs
r0 = 1 / (partial_derivative_vds + 1e-12)

# Limit r0 to 10 MΩ
r0_masked = ma.masked_where((r0 > 1e6), r0)

Av = gm * r0
Av_dB = 20 * np.log10(np.abs(Av))
Av_dB_masked = ma.masked_where((Av_dB < 0) | (Av_dB > 60), Av_dB)

# Meshgrid for plotting
vgs_mesh, vds_mesh = np.meshgrid(vgs_values, vds_values)

# Transpose Av_dB_masked to match the shape of the meshgrid
Av_dB_masked_reshaped = Av_dB_masked.T

# Transpose pivot_table.values to match the meshgrid
IDS_reshaped = (1000 * pivot_table.values).T  # Converting IDS to mA

# Create a figure for the 3D plot and 2D pcolormesh plot
fig = plt.figure(figsize=(18, 10))

# 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(vds_mesh, IDS_reshaped, Av_dB_masked_reshaped, cmap='jet')
ax1.set_xlabel('VDS (V)')
ax1.set_ylabel('IDS (mA)')
ax1.set_zlabel('Av (dB)')
ax1.set_title('EPA018A Surface Plot of |Small Signal Gain| Vs Bais Conditions ')
# Adding color bar to the 3D plot
cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))  # Format color bar ticks
cbar.set_label('|Av|(dB)') 

# 2D pcolormesh plot
ax2 = fig.add_subplot(122)
mesh = ax2.pcolormesh(vds_mesh, IDS_reshaped, Av_dB_masked_reshaped, cmap='jet', shading='auto')
ax2.set_xlabel('VDS (V)')
ax2.set_ylabel('IDS (mA)')
ax2.set_title('EPA018A |Small Signal Gain| Vs Bias Conditions')
# Make the 2D plot square
#ax2.set_aspect('equal', 'box')
# Add grid lines
ax2.grid(True)  # Add grid lines
# Adding color bar to the 2D plot
cbar2 = fig.colorbar(mesh, ax=ax2, shrink=0.5, aspect=5)
cbar2.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))  # Format color bar ticks
cbar2.set_label('|Av|(dB)')

# Show the plot
plt.show()
