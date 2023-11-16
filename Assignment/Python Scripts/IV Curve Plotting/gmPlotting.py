import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
excel_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV_Curves_Updated.xlsx'
df = pd.read_excel(excel_file_path)

# Convert columns to numeric values
df['VGS'] = pd.to_numeric(df['VGS'], errors='coerce')
df['VDS'] = pd.to_numeric(df['VDS'], errors='coerce')
df['IDS'] = pd.to_numeric(df['IDS'], errors='coerce')

# Convert IDS from A to mA
df['IDS_mA'] = df['IDS'] * 1000  # Convert A to mA

# Define the specific VDS values we're interested in for the transconductance plots
vds_values_for_gm = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5]

# Create a pivot table for the data
pivot_df = df.pivot_table(values='IDS_mA', index='VGS', columns='VDS')

# Create a figure for the transconductance plot
plt.figure(figsize=(10, 6))

# Generate colors based on the VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(vds_values_for_gm)))

# Temporary lists to store handles and labels for the legend
handles = []
labels = []

# Plot the transconductance curves for each VDS value or the closest one available
for desired_vds, color in zip(vds_values_for_gm, color_list):
    # Find the closest VDS column if the desired one is not available
    closest_vds = min(pivot_df.columns, key=lambda x: abs(x - desired_vds))
    
    # Extract the IDS values for the closest VDS value
    ids_for_vds = pivot_df[closest_vds].dropna()
    transconductance = np.gradient(ids_for_vds, ids_for_vds.index)
    
    # Label for the plot
    label = f'VDS = {desired_vds}V' if closest_vds == desired_vds else f'VDS ≈ {closest_vds:.2f}V'
    
    # Plot the transconductance curve
    handle, = plt.plot(ids_for_vds.index, transconductance, label=label, color=color)
    
    # Append handle and label to the lists
    handles.append(handle)
    labels.append(label)

# Add a reversed legend to the plot
plt.legend(handles[::-1], labels[::-1], title="VDS Values", loc="best")

# Label the axes and add title
plt.xlabel('VGS (V)')
plt.ylabel('Gm (mS)')
plt.title('Transconductance vs Gate Voltage')
plt.grid(True)
plt.show()
