# Let's read the content of the file to understand its structure
file_path = './IV Curves Data.txt'

# Read the first few lines to get an idea of the data format
with open(file_path, 'r') as file:
    lines = file.readlines()

# Display the first 10 lines to understand the structure of the data
lines[:10]


#-------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv(file_path, sep='\t', header=0)

# Renaming the columns for easier access (removing special characters and dots)
df.columns = ['VGS', 'VDS', 'IDS']

# Checking the first few rows of the dataframe to ensure it's loaded correctly
df.head()

#---------------------------------------------------------------------------

# Convert the columns to numeric values if they are not already
df['VGS'] = pd.to_numeric(df['VGS'], errors='coerce')
df['VDS'] = pd.to_numeric(df['VDS'], errors='coerce')
df['IDS'] = pd.to_numeric(df['IDS'], errors='coerce')

# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))

# Group the data by VGS and plot each group
for vgs, group in df.groupby('VGS'):
    plt.plot(group['VDS'], 1000*group['IDS'], label=f'VGS = {vgs:.2f} V')

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('IDS (mA)')
plt.title('IV Curves of the Transistor')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#-------------------------------------------------
# Re-plotting the IV curves using the previously loaded data

# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))

# Group the data by VGS and plot each group
for vgs, group in df.groupby('VGS'):
    plt.plot(group['VDS'], group['IDS'], label=f'VGS = {vgs:.2f} V')

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('IDS (A)')
plt.title('IV Curves of the Transistor')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#------------------------------------------------
# Convert IDS from A to mA
df['IDS_mA'] = df['IDS'] * 1000

# Re-plotting the IV curves with the y-axis in mA
plt.figure(figsize=(10, 6))

# Group the data by VGS and plot each group
for vgs, group in df.groupby('VGS'):
    plt.plot(group['VDS'], group['IDS_mA'], label=f'VGS = {vgs:.2f} V')

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('IDS (mA)')  # Now in mA
plt.title('EPA018A IV Curves')  # Updated title

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#-----------------------------------------------------
# Re-plotting the IV curves using a colormap to represent the different values of VGS
import numpy as np

# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))

# Get a colormap reference and generate colors based on the number of unique VGS values
colormap = plt.cm.jet  # You can choose other colormaps like plt.cm.viridis, plt.cm.rainbow, etc.
color_list = [colormap(i) for i in np.linspace(0, 1, len(df['VGS'].unique()))]

# Group the data by VGS and plot each group with a unique color from the colormap
for (vgs, group), color in zip(df.groupby('VGS'), color_list):
    plt.plot(group['VDS'], group['IDS_mA'], label=f'VGS = {vgs:.2f} V', color=color)

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('IDS (mA)')  # Still in mA
plt.title('EPA018A IV Curves')  # Title remains the same

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()


#------------------------------------------------------
# Sorting the groups by VGS so that 0 V comes first and -1.2 V comes last in the legend
sorted_groups = df.groupby('VGS', sort=False).apply(lambda x: x.name).sort_values()
sorted_groups = sorted_groups[::-1] if sorted_groups.iloc[-1] == 0 else sorted_groups

# Re-plotting the IV curves with the sorted legend
plt.figure(figsize=(10, 6))

# Apply the sorted order to the colormap
color_list = [colormap(i) for i in np.linspace(0, 1, len(sorted_groups))]

# Plot each group with a unique color from the colormap
for vgs_value, color in zip(sorted_groups, color_list):
    group = df[df['VGS'] == vgs_value]
    plt.plot(group['VDS'], group['IDS_mA'], label=f'VGS = {vgs_value:.2f} V', color=color)

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('IDS (mA)')
plt.title('EPA018A IV Curves')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#-----------------------------------------------------------
# To calculate the transconductance, we need to take the first derivative of IDS with respect to VDS for each VGS. (THIS IS WRONG)
# This can be done using numpy's gradient function which computes the gradient of an N-dimensional array.

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Loop over each group, calculate the transconductance, and plot
for vgs, group in df.groupby('VGS'):
    # Calculate the transconductance (dI/dV) and convert from A/V to mA/V for plotting
    transconductance = np.gradient(group['IDS_mA'], group['VDS'])
    
    # Plotting transconductance vs VDS
    plt.plot(group['VDS'], transconductance, label=f'VGS = {vgs:.2f} V')

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('Transconductance (mA/V)')
plt.title('Transconductance for Different VGS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#------------------------------------------------------------
# To maintain the same color scheme as before and flip the legend entries, we need to sort the groups accordingly
# and use the same color list. We also need to convert the transconductance from mA/V to mS (millisiemens)
# since 1 S = 1 A/V and therefore 1 mS = 1 mA/V.

# Sort the groups by VGS so that 0 V comes first and -1.2 V comes last in the legend
sorted_groups = df.groupby('VGS', sort=False).apply(lambda x: x.name).sort_values()
sorted_groups = sorted_groups[::-1] if sorted_groups.iloc[-1] == 0 else sorted_groups

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Apply the sorted order to the colormap
color_list = [colormap(i) for i in np.linspace(0, 1, len(sorted_groups))]

# Plot each group with a unique color from the colormap
for vgs_value, color in zip(sorted_groups, color_list):
    group = df[df['VGS'] == vgs_value]
    # Calculate the transconductance G_m (dI/dV) and convert from A/V to mS for plotting
    G_m = np.gradient(group['IDS_mA'], group['VDS'])  # Already in mA/V which is equivalent to mS
    
    # Plotting G_m vs VDS
    plt.plot(group['VDS'], G_m, label=f'VGS = {vgs_value:.2f} V', color=color)

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('G_m (mS)')  # Transconductance in mS
plt.title('Transconductance (G_m) for Different VGS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#-----------------------------------------------------------
# Updating the y-axis label to use "Gm" instead of "G_m" for the transconductance unit

# Re-creating the figure for the transconductance plot with the updated label
plt.figure(figsize=(10, 6))

# Plot each group with a unique color from the colormap
for vgs_value, color in zip(sorted_groups, color_list):
    group = df[df['VGS'] == vgs_value]
    # Calculate the transconductance Gm (dI/dV) and convert from A/V to mS for plotting
    Gm = np.gradient(group['IDS_mA'], group['VDS'])  # Already in mA/V which is equivalent to mS
    
    # Plotting Gm vs VDS
    plt.plot(group['VDS'], Gm, label=f'VGS = {vgs_value:.2f} V', color=color)

# Label the axes
plt.xlabel('VDS (V)')
plt.ylabel('Gm (mS)')  # Transconductance in mS with updated label
plt.title('Transconductance (Gm) for Different VGS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#----------------------------------------------
# To properly calculate the transconductance as the partial derivative of IDS with respect to VGS, we will group the data by VDS.
# Then for each group, we will calculate the derivative of IDS with respect to VGS.

# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))

# Initialize an empty dictionary to hold the transconductance data
transconductance_data = {}

# We need to sort the data by VGS to ensure the derivatives are calculated correctly
df_sorted_by_VGS = df.sort_values(by='VGS')

# Group the data by VDS and calculate the transconductance for each group
for vds, group in df_sorted_by_VGS.groupby('VDS'):
    # Only calculate if we have at least two points to perform a derivative
    if group.shape[0] >= 2:
        # Calculate the transconductance (dI/dV) and convert from A/V to mS for plotting
        transconductance = np.gradient(group['IDS_mA'], group['VGS'])
        transconductance_data[vds] = (group['VGS'].values, transconductance)

# We will now plot using the same color scheme as before, with the sorted legend entries.
# Get the sorted VDS values
sorted_VDS_values = sorted(transconductance_data.keys(), reverse=True)

# Generate colors based on the number of unique VDS values
color_list = [colormap(i) for i in np.linspace(0, 1, len(sorted_VDS_values))]

# Plot the transconductance curves
for vds, color in zip(sorted_VDS_values, color_list):
    vgs_values, gm_values = transconductance_data[vds]
    plt.plot(vgs_values, gm_values, label=f'VDS = {vds:.2f} V', color=color)

# Label the axes
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')  # Transconductance in mS
plt.title('Transconductance (Gm) for Different VDS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#----------------------------------------------------
# To plot the transconductance for specific values of VDS, we will filter the data accordingly.

# Define the specific VDS values we're interested in
vds_values_to_plot = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]

# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))

# Generate colors based on the number of specified VDS values
color_list = [colormap(i) for i in np.linspace(0, 1, len(vds_values_to_plot))]

# Plot the transconductance curves for each specified VDS
for vds, color in zip(vds_values_to_plot, color_list):
    # Filter the data for the current VDS value
    subset = df_sorted_by_VGS[df_sorted_by_VGS['VDS'] == vds]
    
    # Only calculate if we have at least two points to perform a derivative
    if subset.shape[0] >= 2:
        # Calculate the transconductance (dI/dV) and convert from A/V to mS for plotting
        transconductance = np.gradient(subset['IDS_mA'], subset['VGS'])
        plt.plot(subset['VGS'], transconductance, label=f'VDS = {vds:.2f} V', color=color)

# Label the axes
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')  # Transconductance in mS
plt.title('Transconductance (Gm) for Selected VDS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#---------------------------------
# It seems there was an issue with filtering the dataset for the specified VDS values. Let's ensure the dataset contains the required VDS values and plot them accordingly.

# Filter the DataFrame to include only the specified VDS values
filtered_df = df[df['VDS'].isin(vds_values_to_plot)]

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Plot the transconductance curves for each specified VDS
for vds_value, color in zip(sorted(vds_values_to_plot), color_list):
    # Filter the data for the current VDS value
    vds_subset = filtered_df[filtered_df['VDS'] == vds_value]
    
    # Sort the data by VGS to ensure the derivatives are calculated correctly
    vds_subset = vds_subset.sort_values(by='VGS')
    
    # Only proceed if we have at least two points
    if len(vds_subset) >= 2:
        # Calculate the transconductance (dI/dV)
        Gm = np.gradient(vds_subset['IDS_mA'], vds_subset['VGS'])
        
        # Plotting Gm vs VGS
        plt.plot(vds_subset['VGS'], Gm, label=f'VDS = {vds_value} V', color=color)

# Label the axes
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('Transconductance (Gm) for Selected VDS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()


#-------------------------------
#Uploaded IV Curves for Gm.txt here
#-------------------------------
# Let's first load the new data from the provided text file to examine its structure and content.

# Load the data from the new file
new_data_path = './IV Curves Data for Gm.txt'
new_df = pd.read_csv(new_data_path, sep="\t")

# Display the first few rows of the dataframe to understand its structure
new_df.head()

#------------------------------------
# Clean up the data by converting columns to numeric values and renaming the columns for easier access

# Convert columns to numeric values
new_df['VGS'] = pd.to_numeric(new_df['VGS'], errors='coerce')
new_df['VDS'] = pd.to_numeric(new_df['VDS'], errors='coerce')
new_df['IDS'] = pd.to_numeric(new_df['fetCharTestingSmartSim0..IDS.i[0, ::]'], errors='coerce')

# Drop the original IDS column with the long name
new_df = new_df.drop(columns=['fetCharTestingSmartSim0..IDS.i[0, ::]'])

# Rename the IDS column to 'IDS_mA', assuming the values are in Amperes and we want to convert them to milliAmperes
new_df['IDS_mA'] = new_df['IDS'] * 1000  # Convert A to mA

# Define the specific VDS values we're interested in for the transconductance plots
vds_values_for_gm = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]

# Filter the DataFrame to include only the specified VDS values
filtered_gm_df = new_df[new_df['VDS'].isin(vds_values_for_gm)]

# Verify the filtered data
filtered_gm_df.head()

#-----------------------------------------------------
# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Plot the transconductance curves for each specified VDS
for vds_value in sorted(vds_values_for_gm):
    # Filter the data for the current VDS value
    vds_subset = filtered_gm_df[filtered_gm_df['VDS'] == vds_value].sort_values(by='VGS')
    
    # Only proceed if we have at least two points
    if len(vds_subset) >= 2:
        # Calculate the transconductance (dI/dV)
        Gm = np.gradient(vds_subset['IDS_mA'], vds_subset['VGS'])
        
        # Plotting Gm vs VGS
        plt.plot(vds_subset['VGS'], Gm, label=f'VDS = {vds_value} V')

# Label the axes
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('Transconductance (Gm) for Selected VDS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#----------------------------------------------
# Let's investigate why some VDS curves are missing.
# It's possible that the filtering did not work as expected due to the precision of floating-point representation.
# We will recheck the VDS values present in the data and ensure that the filtering captures all the required VDS values.

# Get the unique VDS values in the filtered data to see which ones are present
unique_vds_in_filtered_data = filtered_gm_df['VDS'].unique()

# Display the unique VDS values
unique_vds_in_filtered_data
#--------------------------------------------------------
# It appears that the problem might be due to floating-point precision issues when comparing the VDS values.
# To resolve this, we can round the VDS values to two decimal places and then perform the filtering.

# Round the VDS column to two decimal places to avoid floating-point precision issues
filtered_gm_df['VDS_rounded'] = filtered_gm_df['VDS'].round(2)

# Now let's attempt to filter the DataFrame again using the rounded VDS values
filtered_gm_df = filtered_gm_df[filtered_gm_df['VDS_rounded'].isin(vds_values_for_gm)]

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Plot the transconductance curves for each specified VDS
for vds_value in sorted(vds_values_for_gm):
    # Filter the data for the current VDS value using the rounded column
    vds_subset = filtered_gm_df[filtered_gm_df['VDS_rounded'] == vds_value].sort_values(by='VGS')
    
    # Only proceed if we have at least two points
    if len(vds_subset) >= 2:
        # Calculate the transconductance (dI/dV)
        Gm = np.gradient(vds_subset['IDS_mA'], vds_subset['VGS'])
        
        # Plotting Gm vs VGS
        plt.plot(vds_subset['VGS'], Gm, label=f'VDS = {vds_value} V')

# Label the axes
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('Transconductance (Gm) for Selected VDS Values')

# Add a legend to the plot
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

#-----------------------------------------------
# To plot the matrix in 3D, we'll create a pivot table from the dataframe where:
# - VDS values will be along the x-axis
# - VGS values will be along the y-axis
# - IDS values will be the z-axis (heights)

# Create a pivot table for the data
pivot_df = new_df.pivot_table(values='IDS_mA', index='VGS', columns='VDS')

# Now we can create a 3D plot using this pivot table
from mpl_toolkits.mplot3d import Axes3D

# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare the grid for plotting
X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
Z = pivot_df.values

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Label the axes
ax.set_xlabel('VDS (V)')
ax.set_ylabel('VGS (V)')
ax.set_zlabel('IDS (mA)')

# Add a color bar to the plot
fig.colorbar(surf)

# Show the plot
plt.show()

#---------------------------------------
# To plot the transconductance as the partial derivative of IDS with respect to VGS at VDS = 3V,
# we need to take a slice of the pivot table at VDS = 3V and then calculate the derivative.

# First, let's check if VDS = 3V is available in our pivot table
if 3 in pivot_df.columns:
    # Extract the slice for VDS = 3V
    ids_at_vds_3 = pivot_df[3]
    
    # Calculate the transconductance, which is the gradient of IDS with respect to VGS
    vgs_values = pivot_df.index
    transconductance_at_vds_3 = np.gradient(ids_at_vds_3, vgs_values)
    
    # Now let's plot the transconductance
    plt.figure(figsize=(10, 6))
    plt.plot(vgs_values, transconductance_at_vds_3, label='VDS = 3V')
    
    # Label the axes and add title and legend
    plt.xlabel('VGS (V)')
    plt.ylabel('Transconductance (mS)')
    plt.title('Transconductance at VDS = 3V')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("VDS = 3V is not available in the dataset.")


#----------------------------------------
# To find the closest VDS value to 3V, we can subtract 3 from all VDS values, take the absolute value, and then find the minimum.

# Get the absolute difference from 3 for all VDS values in the pivot table
closest_vds_value = (np.abs(pivot_df.columns - 3)).argmin()

# Now get the VDS value that is closest to 3V
vds_closest_to_3 = pivot_df.columns[closest_vds_value]

# Extract the slice for the VDS value closest to 3V
ids_at_closest_vds = pivot_df[vds_closest_to_3]

# Calculate the transconductance for this VDS
transconductance_at_closest_vds = np.gradient(ids_at_closest_vds, vgs_values)

# Plot the transconductance
plt.figure(figsize=(10, 6))
plt.plot(vgs_values, transconductance_at_closest_vds, label=f'VDS = {vds_closest_to_3}V')

# Label the axes and add title and legend
plt.xlabel('VGS (V)')
plt.ylabel('Transconductance (mS)')
plt.title(f'Transconductance at VDS ≈ 3V (Actual VDS = {vds_closest_to_3}V)')
plt.legend()
plt.grid(True)
plt.show()

#----------------------------------------------------------
# Let's first find the closest column to VDS = 3V in the pivot table and then extract the corresponding IDS values.
# We will also ensure that the VGS values match the length of the IDS values before calculating the gradient.

# Find the column in the pivot table closest to VDS = 3V
closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - 3)).argmin()]

# Extract the IDS values for the closest VDS column
ids_for_closest_vds = pivot_df[closest_vds_column].values

# Make sure we only take VGS values where IDS values are not NaN
vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]

# Calculate the transconductance (partial derivative of IDS with respect to VGS)
transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)

# Plot the transconductance
plt.figure(figsize=(10, 6))
plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column}V')

# Label the axes and add title and legend
plt.xlabel('VGS (V)')
plt.ylabel('Transconductance (mS)')
plt.title(f'Transconductance at VDS ≈ 3V (Actual VDS = {closest_vds_column}V)')
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------------
# We will now add curves to the transconductance plot for VDS values as close as possible to
# [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3] V.

# Define the VDS values for which we want to plot the transconductance curves
desired_vds_values = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# For each desired VDS value, find the closest actual VDS value in the data and plot the transconductance
for desired_vds in desired_vds_values:
    # Find the column in the pivot table closest to the desired VDS value
    closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - desired_vds)).argmin()]
    
    # Extract the IDS values for the closest VDS column
    ids_for_closest_vds = pivot_df[closest_vds_column].values
    
    # Make sure we only take VGS values where IDS values are not NaN
    vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
    ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]
    
    # Calculate the transconductance
    transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)
    
    # Plot the transconductance curve
    plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column}V')

# Label the axes and add title and legend
plt.xlabel('VGS (V)')
plt.ylabel('Transconductance (mS)')
plt.title('Transconductance (Gm) for VDS Values Close to Desired')
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------
# We will apply a rainbow color scheme to the plot and ensure the legend entries are rounded to two decimal points.

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the number of desired VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(desired_vds_values)))

# For each desired VDS value, find the closest actual VDS value in the data and plot the transconductance
for desired_vds, color in zip(desired_vds_values, color_list):
    # Find the column in the pivot table closest to the desired VDS value
    closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - desired_vds)).argmin()]
    
    # Extract the IDS values for the closest VDS column
    ids_for_closest_vds = pivot_df[closest_vds_column].values
    
    # Make sure we only take VGS values where IDS values are not NaN
    vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
    ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]
    
    # Calculate the transconductance
    transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)
    
    # Plot the transconductance curve
    plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column:.2f}V', color=color)

# Label the axes and add title
plt.xlabel('VGS (V)')
plt.ylabel('Transconductance (mS)')
plt.title('Transconductance (Gm) for VDS Values Close to Desired')

# Add a legend to the plot with rounded VDS values
plt.legend(title="VDS Values", loc="best")

# Show the plot with a grid
plt.grid(True)
plt.show()

#------------------------------------------------
# We will flip the order of the legend entries, change the title to "EPA018A Transconductance vs Gate Voltage",
# and adjust the y-axis label to "Gm (mS)".

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the number of desired VDS values, reversed for the legend
color_list = plt.cm.rainbow(np.linspace(0, 1, len(desired_vds_values)))[::-1]

# For each desired VDS value, find the closest actual VDS value in the data and plot the transconductance
for desired_vds, color in zip(reversed(desired_vds_values), color_list):
    # Find the column in the pivot table closest to the desired VDS value
    closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - desired_vds)).argmin()]
    
    # Extract the IDS values for the closest VDS column
    ids_for_closest_vds = pivot_df[closest_vds_column].values
    
    # Make sure we only take VGS values where IDS values are not NaN
    vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
    ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]
    
    # Calculate the transconductance
    transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)
    
    # Plot the transconductance curve
    plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column:.2f}V', color=color)

# Label the axes and add title
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('EPA018A Transconductance vs Gate Voltage')

# Add a legend to the plot with reversed entries
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title="VDS Values", loc="best")

# Show the plot with a grid
plt.grid(True)
plt.show()

#-------------------------------------------
# We will rearrange the legend so that VDS=3V is the first entry and VDS=0.25V is the final one.

# To do this, we will plot in order of VDS values starting from 3V down to 0.25V.

# Sort the desired VDS values in descending order for plotting
sorted_vds_values = sorted(desired_vds_values, reverse=True)

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the number of sorted VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(sorted_vds_values)))

# Plot the transconductance curves for each VDS value in the sorted list
for desired_vds, color in zip(sorted_vds_values, color_list):
    # Find the column in the pivot table closest to the desired VDS value
    closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - desired_vds)).argmin()]
    
    # Extract the IDS values for the closest VDS column
    ids_for_closest_vds = pivot_df[closest_vds_column].values
    
    # Make sure we only take VGS values where IDS values are not NaN
    vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
    ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]
    
    # Calculate the transconductance
    transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)
    
    # Plot the transconductance curve
    plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column:.2f}V', color=color)

# Label the axes and add title
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('EPA018A Transconductance vs Gate Voltage')

# Add a legend to the plot with the specified order
plt.legend(title="VDS Values", loc="best")

# Show the plot with a grid
plt.grid(True)
plt.show()


#---------------------------------------------
# We will add curves for VDS = 4V and 5V to the transconductance plot, if available in the data.

# Add the new desired VDS values
extended_vds_values = sorted_vds_values + [4, 5]

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the number of extended VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(extended_vds_values)))

# Plot the transconductance curves for each VDS value in the extended list
for desired_vds, color in zip(extended_vds_values, color_list):
    # Check if the desired VDS is in the data
    if desired_vds in pivot_df.columns:
        # Extract the IDS values for the VDS value
        ids_for_vds = pivot_df[desired_vds].values

        # Make sure we only take VGS values where IDS values are not NaN
        vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_vds)].values
        ids_for_vds_nonan = ids_for_vds[~np.isnan(ids_for_vds)]

        # Calculate the transconductance
        transconductance = np.gradient(ids_for_vds_nonan, vgs_values_nonan)

        # Plot the transconductance curve
        plt.plot(vgs_values_nonan, transconductance, label=f'VDS = {desired_vds}V', color=color)
    else:
        # If the desired VDS is not in the data, find the closest column
        closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - desired_vds)).argmin()]
        
        # Extract the IDS values for the closest VDS column
        ids_for_closest_vds = pivot_df[closest_vds_column].values
        
        # Make sure we only take VGS values where IDS values are not NaN
        vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
        ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]
        
        # Calculate the transconductance
        transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)

        # Plot the transconductance curve for the closest VDS value
        plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column:.2f}V', color=color)

# Label the axes and add title
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('EPA018A Transconductance vs Gate Voltage')

# Add a legend to the plot with the specified order
plt.legend(title="VDS Values", loc="best")

# Show the plot with a grid
plt.grid(True)
plt.show()


#----------------------------------------
# We will re-plot the transconductance curves, ensuring that the 5V curve is at the top of the legend, followed by 4V, 3V, and so on.

# Define the VDS values in the new order for the legend: 5V at the top, then 4V, then descending from 3V
new_legend_order_vds_values = [5, 4] + sorted(desired_vds_values, reverse=True)

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the new legend order of VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(new_legend_order_vds_values)))

# Plot the transconductance curves for each VDS value in the new legend order
for desired_vds, color in zip(new_legend_order_vds_values, color_list):
    # Check if the desired VDS is in the data
    if desired_vds in pivot_df.columns:
        # Extract the IDS values for the VDS value
        ids_for_vds = pivot_df[desired_vds].values

        # Make sure we only take VGS values where IDS values are not NaN
        vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_vds)].values
        ids_for_vds_nonan = ids_for_vds[~np.isnan(ids_for_vds)]

        # Calculate the transconductance
        transconductance = np.gradient(ids_for_vds_nonan, vgs_values_nonan)

        # Plot the transconductance curve
        plt.plot(vgs_values_nonan, transconductance, label=f'VDS = {desired_vds}V', color=color)
    else:
        # If the desired VDS is not in the data, find the closest column
        closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - desired_vds)).argmin()]
        
        # Extract the IDS values for the closest VDS column
        ids_for_closest_vds = pivot_df[closest_vds_column].values
        
        # Make sure we only take VGS values where IDS values are not NaN
        vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
        ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]
        
        # Calculate the transconductance
        transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)

        # Plot the transconductance curve for the closest VDS value
        plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column:.2f}V', color=color)

# Label the axes and add title
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('EPA018A Transconductance vs Gate Voltage')

# Add a legend to the plot with the new order
handles, labels = plt.gca().get_legend_handles_labels()
# Create a mapping from the VDS value (or its approximation) to the handle (line object)
handle_label_mapping = dict(zip(labels, handles))
# Order the handles according to the new_legend_order_vds_values list
ordered_handles = [handle_label_mapping[f'VDS = {v:.2f}V'] if f'VDS = {v:.2f}V' in handle_label_mapping 
                   else handle_label_mapping[f'VDS ≈ {v:.2f}V'] for v in new_legend_order_vds_values]
# Re-create the labels list to match the ordered handles
ordered_labels = [f'VDS = {v:.2f}V' if f'VDS = {v:.2f}V' in handle_label_mapping 
                  else f'VDS ≈ {v:.2f}V' for v in new_legend_order_vds_values]

# Add the ordered legend to the plot
plt.legend(ordered_handles, ordered_labels, title="VDS Values", loc="best")

# Show the plot with a grid
plt.grid(True)
plt.show()


#-----------------------------------
# We will re-plot the transconductance curves without the 4V curve.

# Remove the 4V value from the list
new_legend_order_vds_values.remove(4)

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the new legend order of VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(new_legend_order_vds_values)))

# Plot the transconductance curves for each VDS value in the new legend order
for desired_vds, color in zip(new_legend_order_vds_values, color_list):
    # Check if the desired VDS is in the data
    if desired_vds in pivot_df.columns:
        # Extract the IDS values for the VDS value
        ids_for_vds = pivot_df[desired_vds].values

        # Make sure we only take VGS values where IDS values are not NaN
        vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_vds)].values
        ids_for_vds_nonan = ids_for_vds[~np.isnan(ids_for_vds)]

        # Calculate the transconductance
        transconductance = np.gradient(ids_for_vds_nonan, vgs_values_nonan)

        # Plot the transconductance curve
        plt.plot(vgs_values_nonan, transconductance, label=f'VDS = {desired_vds}V', color=color)
    else:
        # If the desired VDS is not in the data, find the closest column
        closest_vds_column = pivot_df.columns[(np.abs(pivot_df.columns - desired_vds)).argmin()]
        
        # Extract the IDS values for the closest VDS column
        ids_for_closest_vds = pivot_df[closest_vds_column].values
        
        # Make sure we only take VGS values where IDS values are not NaN
        vgs_values_nonan = pivot_df.index[~np.isnan(ids_for_closest_vds)].values
        ids_for_closest_vds_nonan = ids_for_closest_vds[~np.isnan(ids_for_closest_vds)]
        
        # Calculate the transconductance
        transconductance = np.gradient(ids_for_closest_vds_nonan, vgs_values_nonan)

        # Plot the transconductance curve for the closest VDS value
        plt.plot(vgs_values_nonan, transconductance, label=f'VDS ≈ {closest_vds_column:.2f}V', color=color)

# Label the axes and add title
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('EPA018A Transconductance vs Gate Voltage')

# Add a legend to the plot with the new order, excluding the 4V curve
handles, labels = plt.gca().get_legend_handles_labels()
# Filter out the 4V handle and label
handles_and_labels = zip(handles, labels)
filtered_handles_and_labels = [(h, l) for h, l in handles_and_labels if '4.00' not in l]
ordered_handles, ordered_labels = zip(*filtered_handles_and_labels)

# Add the ordered legend to the plot
plt.legend(ordered_handles, ordered_labels, title="VDS Values", loc="best")

# Show the plot with a grid
plt.grid(True)
plt.show()
