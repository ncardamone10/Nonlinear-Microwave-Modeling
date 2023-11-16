# This doesn't work

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the new file
# #new_data_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV Curves Data for Gm.txt'
# new_data_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV Curves Updated.txt'

# new_df = pd.read_csv(new_data_path, sep="\t")

# # Convert columns to numeric values and handle errors by coercing to NaN
# new_df['VGS'] = pd.to_numeric(new_df['VGS'], errors='coerce')
# new_df['VDS'] = pd.to_numeric(new_df['VDS'], errors='coerce')
# new_df['IDS'] = pd.to_numeric(new_df['fetCharTestingSmartSim0..IDS.i[0, ::]'], errors='coerce')

# # Drop the original IDS column with the long name
# new_df = new_df.drop(columns=['fetCharTestingSmartSim0..IDS.i[0, ::]'])

# # Convert IDS from A to mA
# new_df['IDS_mA'] = new_df['IDS'] * 1000

# # Define the specific VDS values we're interested in for the plots
# vds_values_for_gm = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5]

# # Create a pivot table for the data
# pivot_df = new_df.pivot_table(values='IDS_mA', index='VGS', columns='VDS')

# # Create a subplot layout with 2 rows
# fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# # Generate colors based on the VDS values
# color_list = plt.cm.rainbow(np.linspace(0, 1, len(vds_values_for_gm)))

# # Plot both the IV curves and the transconductance curves within the same for loop
# for desired_vds, color in zip(vds_values_for_gm, color_list):
#     # Check if the desired VDS is in the data
#     if desired_vds in pivot_df.columns:
#         # Extract the IDS values for the VDS value
#         ids_for_vds = pivot_df[desired_vds].values
#         vgs_values = pivot_df.index[~np.isnan(ids_for_vds)]
#         ids_for_vds_nonan = ids_for_vds[~np.isnan(ids_for_vds)]

#         # Calculate the transconductance
#         transconductance = np.gradient(ids_for_vds_nonan, vgs_values)

#         # Plot IV curve in the top subplot
#         axs[0].plot(vgs_values, ids_for_vds_nonan, label=f'VDS = {desired_vds}V', color=color)

#         # Plot the transconductance curve in the bottom subplot
#         axs[1].plot(vgs_values, transconductance, label=f'VDS = {desired_vds}V', color=color)

# # Configure the top subplot (IV curves)
# axs[0].set_xlabel('VGS (V)')
# axs[0].set_ylabel('IDS (mA)')
# axs[0].set_title('IV Curves')
# axs[0].legend(title="VDS Values", loc="upper right")
# axs[0].grid(True)

# # Configure the bottom subplot (Transconductance curves)
# axs[1].set_xlabel('VGS (V)')
# axs[1].set_ylabel('Transconductance (mS)')
# axs[1].set_title('Transconductance vs Gate Voltage')
# axs[1].legend(title="VDS Values", loc="upper right")
# axs[1].grid(True)

# # Adjust the layout
# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
excel_data_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV_Curves_Updated.xlsx'

# Read the data from the Excel file
df = pd.read_excel(excel_data_path)

# Convert columns to numeric values (this should be straightforward as they should already be in numeric format)
df['VGS'] = pd.to_numeric(df['VGS'], errors='coerce')
df['VDS'] = pd.to_numeric(df['VDS'], errors='coerce')
df['IDS'] = pd.to_numeric(df['IDS'], errors='coerce')

# Convert IDS from A to mA
df['IDS_mA'] = df['IDS'] * 1000

# Define the specific VDS values we're interested in for the plots
vds_values_for_gm = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5]

# Create a pivot table for the data
pivot_df = df.pivot_table(values='IDS_mA', index='VGS', columns='VDS')

# Create a subplot layout with 2 rows
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Generate colors based on the VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(vds_values_for_gm)))

# Plot both the IV curves and the transconductance curves within the same for loop
for desired_vds, color in zip(vds_values_for_gm, color_list):
    # Check if the desired VDS is in the data
    if desired_vds in pivot_df.columns:
        # Extract the IDS values for the VDS value
        ids_for_vds = pivot_df[desired_vds].values
        vgs_values = pivot_df.index[~np.isnan(ids_for_vds)]
        ids_for_vds_nonan = ids_for_vds[~np.isnan(ids_for_vds)]

        # Calculate the transconductance
        transconductance = np.gradient(ids_for_vds_nonan, vgs_values)

        # Plot IV curve in the top subplot
        axs[0].plot(vgs_values, ids_for_vds_nonan, label=f'VDS = {desired_vds}V', color=color)

        # Plot the transconductance curve in the bottom subplot
        axs[1].plot(vgs_values, transconductance, label=f'VDS = {desired_vds}V', color=color)

# Configure the top subplot (IV curves)
axs[0].set_xlabel('VGS (V)')
axs[0].set_ylabel('IDS (mA)')
axs[0].set_title('IV Curves')
axs[0].legend(title="VDS Values", loc="upper right")
axs[0].grid(True)

# Configure the bottom subplot (Transconductance curves)
axs[1].set_xlabel('VGS (V)')
axs[1].set_ylabel('Transconductance (mS)')
axs[1].set_title('Transconductance vs Gate Voltage')
axs[1].legend(title="VDS Values", loc="upper right")
axs[1].grid(True)

# Adjust the layout
plt.tight_layout()
plt.show()

