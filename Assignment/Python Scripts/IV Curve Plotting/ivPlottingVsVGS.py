import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the provided Excel file
excel_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV_Curves_Updated.xlsx'
df = pd.read_excel(excel_file_path)

# Dropping rows with missing values (if any)
df.dropna(inplace=True)

# Converting data to numeric
df['VGS'] = pd.to_numeric(df['VGS'], errors='coerce')
df['VDS'] = pd.to_numeric(df['VDS'], errors='coerce')
df['IDS'] = pd.to_numeric(df['IDS'], errors='coerce')

# Pivot the DataFrame to get IDS values for each combination of VGS and VDS
pivot_df = df.pivot_table(values='IDS', index='VGS', columns='VDS')

# List of desired VDS values
desired_vds_values = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 3, 5]

# Find the closest VDS values present in the data for this new list
available_vds_values = pivot_df.columns
closest_vds_values = [min(available_vds_values, key=lambda x: abs(x - vds)) for vds in desired_vds_values]

# Generate colors based on the number of VDS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(closest_vds_values)))

# Plotting the IV curves with rainbow color scheme
plt.figure(figsize=(12, 8))

for vds, color in zip(closest_vds_values, color_list):
    plt.plot(pivot_df.index, pivot_df[vds] * 1000, label=f'VDS = {vds:.2f}V', color=color)  # Converting A to mA

plt.xlabel('VGS (V)')
plt.ylabel('IDS (mA)')
plt.title('EPA018A IV Curves vs VGS')

# Reversing the order of legend entries
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title="VDS Values", loc="best")

plt.grid(True)
plt.show()
