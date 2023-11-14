
# This plots IDS as a function of VDS and VGS (3D surface)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the data from the file
file_path = './IV Curves Data for Gm.txt'
data = pd.read_csv(file_path, sep='\t', engine='python')

# Rename the 'IDS' column and convert all columns to numeric, handling non-numeric values as NaN
data.rename(columns={'fetCharTestingSmartSim0..IDS.i[0, ::]': 'IDS'}, inplace=True)
data['VGS'] = pd.to_numeric(data['VGS'], errors='coerce')
data['VDS'] = pd.to_numeric(data['VDS'], errors='coerce')
data['IDS'] = pd.to_numeric(data['IDS'], errors='coerce')

# Drop rows with NaN values that resulted from the conversion
data.dropna(inplace=True)

# Convert IDS from A to mA for consistency with previous plots
data['IDS_mA'] = data['IDS'] * 1000

# Create a pivot table for the 3D plot with 'VGS' as rows, 'VDS' as columns, and 'IDS_mA' as values
pivot_df = data.pivot_table(values='IDS_mA', index='VGS', columns='VDS')

# Create a meshgrid for the 3D plot
X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
Z = pivot_df.values

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Label the axes and add title
ax.set_xlabel('VDS (V)')
ax.set_ylabel('VGS (V)')
ax.set_zlabel('IDS (mA)')
ax.set_title('3D Surface Plot of IV Characteristics')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the plot
plt.show()




# #------------------------------------------
# # Also plots everything (same as section below, but with out the masking of data)
# # Not really sure what the point of this section is
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Read the data from the file
# file_path = './IV Curves Data for Gm.txt'  # Make sure to use the correct path for your file
# data = pd.read_csv(file_path, sep='\t', engine='python')

# # Convert the 'IDS' column to numeric values
# data.rename(columns={'fetCharTestingSmartSim0..IDS.i[0, ::]': 'IDS'}, inplace=True)
# data['VGS'] = pd.to_numeric(data['VGS'], errors='coerce')
# data['VDS'] = pd.to_numeric(data['VDS'], errors='coerce')
# data['IDS'] = pd.to_numeric(data['IDS'], errors='coerce')
# data.dropna(inplace=True)

# # Create a pivot table for VGS and VDS
# pivot_table = data.pivot_table(values='IDS', index='VGS', columns='VDS')

# # Calculate the partial derivative of IDS with respect to VGS
# partial_derivative_vgs = np.gradient(pivot_table.values, axis=0)

# # Calculate the partial derivative of IDS with respect to VDS
# partial_derivative_vds = np.gradient(pivot_table.values, axis=1)

# gm = partial_derivative_vgs
# r0 = 1/partial_derivative_vds

# Av = gm*r0

# # Create a meshgrid for plotting using the unique values of VGS and VDS from the pivot table
# vgs_values, vds_values = np.meshgrid(pivot_table.columns, pivot_table.index)



# # Create a figure to hold the subplots
# fig = plt.figure(figsize=(14, 10))

# # First subplot -- IV Curves
# ax1 = fig.add_subplot(221, projection='3d')
# surf1 = ax1.plot_surface(vgs_values, vds_values, 1000*pivot_table.values, cmap='jet')

# # Label the axes and add title
# ax1.set_xlabel('VDS (V)')
# ax1.set_ylabel('VGS (V)')
# ax1.set_zlabel('IDS (mA)')
# ax1.set_title('EPA018A IV Curve')

# # Add a color bar which maps values to colors
# fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)



# # Second subplot -- Transconductance (partial wrt VGS)
# ax2 = fig.add_subplot(222, projection='3d')
# surf2 = ax2.plot_surface(vgs_values, vds_values, 1000*gm, cmap='jet')

# # Label the axes and add title
# ax2.set_xlabel('VDS (V)')
# ax2.set_ylabel('VGS (V)')
# ax2.set_zlabel('Gm (mS)')
# ax2.set_title('EPA181A Transconductance (Gm)')

# fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# # Third subplot -- r0 (partial wrt vds)^-1
# ax3 = fig.add_subplot(223, projection='3d')
# surf3 = ax3.plot_surface(vgs_values, vds_values, np.log10(r0), cmap='jet')

# # Label the axes and add title
# ax3.set_xlabel('VDS (V)')
# ax3.set_ylabel('VGS (V)')
# ax3.set_zlabel('log10(r0) (Ohms)')
# ax3.set_title('EPA018A Output Resistance (r0)')

# fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# # Fourth subplot -- small signal gain
# ax4 = fig.add_subplot(224, projection='3d')
# surf4 = ax4.plot_surface(vgs_values, vds_values, Av, cmap='jet')


# # Label the axes and add title
# ax4.set_xlabel('VDS (V)')
# ax4.set_ylabel('VGS (V)')
# ax4.set_zlabel('Av (V/V)')
# ax4.set_title('EPA Small Signal Voltage Gain')

# fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)

# # Adjust the layout
# plt.tight_layout()

# # Show the plot
# plt.show()



# #-------------------------------
# # This plots gm, ro, IDS and Av as a function of VGS and VDS
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import RectBivariateSpline
# import numpy.ma as ma

# # Read the data
# file_path = './IV Curves Data for Gm.txt'  # Update this path to your data file location
# data = pd.read_csv(file_path, sep='\t', engine='python')

# # Convert and clean data
# data.rename(columns={'fetCharTestingSmartSim0..IDS.i[0, ::]': 'IDS'}, inplace=True)
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
# # Convert vgs_values to 2D array for spline evaluation
# vgs_values_2d = vgs_values[:, np.newaxis]
# partial_derivative_vgs = spline.partial_derivative(1, 0)(vgs_values_2d, vds_values)
# partial_derivative_vds = spline.partial_derivative(0, 1)(vgs_values_2d, vds_values)

# gm = partial_derivative_vgs
# r0 = 1 / partial_derivative_vds

# # Limit r0 to 10 MΩ
# # r0[r0 > 10e6] = 10e6

# r0_masked = ma.masked_where((r0 > 1e6), r0)


# Av = gm * r0

# Av_masked = ma.masked_where((Av < 0) | (Av > 1000), Av)


# # Meshgrid for plotting
# vgs_mesh, vds_mesh = np.meshgrid(pivot_table.columns, pivot_table.index)

# # Create a figure to hold the subplots
# fig = plt.figure(figsize=(14, 10))

# # First subplot -- IV Curves
# ax1 = fig.add_subplot(221, projection='3d')
# surf1 = ax1.plot_surface(vgs_mesh, vds_mesh, 1000*pivot_table.values, cmap='jet')
# ax1.set_xlabel('VDS (V)')
# ax1.set_ylabel('VGS (V)')
# ax1.set_zlabel('IDS (mA)')
# ax1.set_title('EPA018A IV Curve')
# fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# # Second subplot -- Transconductance (partial wrt VGS)
# ax2 = fig.add_subplot(222, projection='3d')
# surf2 = ax2.plot_surface(vgs_mesh, vds_mesh, 1000*gm, cmap='jet')
# ax2.set_xlabel('VDS (V)')
# ax2.set_ylabel('VGS (V)')
# ax2.set_zlabel('Gm (mS)')
# ax2.set_title('EPA181A Transconductance (Gm)')
# fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# # Third subplot -- r0 (partial wrt VDS)^-1
# ax3 = fig.add_subplot(223, projection='3d')
# surf3 = ax3.plot_surface(vgs_mesh, vds_mesh, np.log10(r0_masked), cmap='jet')
# ax3.set_xlabel('VDS (V)')
# ax3.set_ylabel('VGS (V)')
# ax3.set_zlabel('log10(r0) (Ohms)')
# ax3.set_title('EPA018A Output Resistance (r0)')
# fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# # Fourth subplot -- small signal gain
# ax4 = fig.add_subplot(224, projection='3d')
# surf4 = ax4.plot_surface(vgs_mesh, vds_mesh, (Av_masked), cmap='jet')
# ax4.set_xlabel('VDS (V)')
# ax4.set_ylabel('VGS (V)')
# ax4.set_zlabel('Av (V/V)')
# ax4.set_title('EPA Small Signal Voltage Gain')
# fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
# ax4.set_zlim(0,1000)

# # Adjust the layout and show the plot
# plt.tight_layout()
# plt.show()


# #-------------------------------------------
# # This plots the low frequency voltage gain as a function of VDS and IDS
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import RectBivariateSpline
# import numpy.ma as ma

# # Read the data
# file_path = './IV Curves Data for Gm.txt'  # Update this path to your data file location
# data = pd.read_csv(file_path, sep='\t', engine='python')

# # Convert and clean data
# data.rename(columns={'fetCharTestingSmartSim0..IDS.i[0, ::]': 'IDS'}, inplace=True)
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
# vgs_mesh, vds_mesh = np.meshgrid(pivot_table.index, pivot_table.columns)

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
