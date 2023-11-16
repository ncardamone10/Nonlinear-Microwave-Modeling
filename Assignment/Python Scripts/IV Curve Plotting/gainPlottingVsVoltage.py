import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
import numpy.ma as ma

# Updated file path
excel_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV_Curves_Updated.xlsx'
data = pd.read_excel(excel_file_path)

# Convert and clean data
data['VGS'] = pd.to_numeric(data['VGS'], errors='coerce')
data['VDS'] = pd.to_numeric(data['VDS'], errors='coerce')
data['IDS'] = pd.to_numeric(data['IDS'], errors='coerce')
data.dropna(inplace=True)

# Create a pivot table for VGS and VDS
pivot_table = data.pivot_table(values='IDS', index='VGS', columns='VDS')

# Bivariate spline approximation
spline = RectBivariateSpline(pivot_table.index, pivot_table.columns, pivot_table)

# Calculate partial derivatives
vgs_values_2d = np.expand_dims(pivot_table.index, axis=1)
partial_derivative_vgs = spline.partial_derivative(1, 0)(vgs_values_2d, pivot_table.columns)
partial_derivative_vds = spline.partial_derivative(0, 1)(vgs_values_2d, pivot_table.columns)

gm = partial_derivative_vgs
ro = 1 / (partial_derivative_vds + 1e-12)  # Output resistance calculation
ro_masked = ma.masked_where((ro > 1e6), ro)  # Limiting ro to 1 MΩ
Av = gm * ro
Av_masked = ma.masked_where((Av < 0) | (Av > 1000), Av)

# Meshgrid for plotting
vgs_mesh, vds_mesh = np.meshgrid(pivot_table.index, pivot_table.columns, indexing='ij')

# Create a figure to hold the subplots
fig = plt.figure(figsize=(14, 10))

# Adding a main title
fig.suptitle("EPA018A 3D Low Frequency IV Curves, Transconductance, Output Resistance and Small Signal Gain", fontsize=16)

# First subplot -- IV Curves
ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(vgs_mesh, vds_mesh, 1000 * pivot_table, cmap='jet')
ax1.set_xlabel('VGS (V)')
ax1.set_ylabel('VDS (V)')
ax1.set_zlabel('IDS (mA)')
ax1.set_title('IV Curve')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Second subplot -- Transconductance (Gm)
ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(vgs_mesh, vds_mesh, 1000 * gm, cmap='jet')
ax2.set_xlabel('VGS (V)')
ax2.set_ylabel('VDS (V)')
ax2.set_zlabel('Gm (mS)')
ax2.set_title('Transconductance (Gm)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# Third subplot -- Output Resistance (ro)
ax3 = fig.add_subplot(223, projection='3d')
surf3 = ax3.plot_surface(vgs_mesh, vds_mesh, np.log10(ro_masked), cmap='jet')
ax3.set_xlabel('VGS (V)')
ax3.set_ylabel('VDS (V)')
ax3.set_zlabel('log10(ro) (Ohms)')
ax3.set_title('Output Resistance (ro)')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# Fourth subplot -- Small Signal Voltage Gain
ax4 = fig.add_subplot(224, projection='3d')
surf4 = ax4.plot_surface(vgs_mesh, vds_mesh, Av_masked, cmap='jet')
ax4.set_xlabel('VGS (V)')
ax4.set_ylabel('VDS (V)')
ax4.set_zlabel('Av (V/V)')
ax4.set_title('Small Signal Voltage Gain')
ax4.set_zlim(0, 1000)
fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)

# Adjust the layout and show the plot
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
