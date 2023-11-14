import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the new file
new_data_path = './IV Curves Data for Gm.txt'
new_df = pd.read_csv(new_data_path, sep="\t")

# Display the first few rows of the dataframe to understand its structure
new_df.head()


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
vds_values_for_gm = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5]


# Create a pivot table for the data
pivot_df = new_df.pivot_table(values='IDS_mA', index='VGS', columns='VDS')


# We will re-plot the transconductance curves, ensuring that the 5V curve is at the top of the legend, followed by 4V, 3V, and so on.

# Define the VDS values in the new order for the legend: 5V at the top, then 4V, then descending from 3V
# new_legend_order_vds_values = [ 4] + sorted(vds_values_for_gm, reverse=True)

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the new legend order of VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(vds_values_for_gm)))


# Create a subplot layout with 2 rows
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot the transconductance curves for each VDS value in the new legend order
for desired_vds, color in zip(vds_values_for_gm, color_list):
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
                   else handle_label_mapping[f'VDS ≈ {v:.2f}V'] for v in vds_values_for_gm]
# Re-create the labels list to match the ordered handles
ordered_labels = [f'VDS = {v:.2f}V' if f'VDS = {v:.2f}V' in handle_label_mapping 
                  else f'VDS ≈ {v:.2f}V' for v in vds_values_for_gm]

# Add the ordered legend to the plot
plt.legend(ordered_handles, ordered_labels, title="VDS Values", loc="best")

# Show the plot with a grid
plt.grid(True)
plt.show()
