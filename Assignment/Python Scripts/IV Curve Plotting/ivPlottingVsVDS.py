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

# List of desired VGS values
desired_vgs_values = [0, -0.2, -0.4, -0.6, -0.8, -1, -1.2]

# Find the closest VGS values present in the data
available_vgs_values = pivot_df.index
closest_vgs_values = [min(available_vgs_values, key=lambda x: abs(x - vgs)) for vgs in desired_vgs_values]

# Generate colors based on the number of VGS values
color_list = plt.cm.rainbow(np.linspace(0, 1, len(closest_vgs_values)))

# Plotting the IV curves for VDS on x-axis and IDS on y-axis
plt.figure(figsize=(12, 8))

for vgs, color in zip(closest_vgs_values, color_list):
    plt.plot(pivot_df.columns, pivot_df.loc[vgs] * 1000, label=f'VGS = {vgs:.2f}V', color=color)  # Converting A to mA

plt.xlabel('VDS (V)')
plt.ylabel('IDS (mA)')
plt.title('IV Curves for Selected VGS Values')
plt.legend(title="VGS Values", loc="best")
plt.grid(True)
plt.show()
