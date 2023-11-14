import pandas as pd
import matplotlib.pyplot as plt





# Replace this with the path to your data file

#file_path = "C:\\Users\\ncard\\OneDrive - University of Ottawa\\University Archive\\Masters\\Nonlinear uWave Modeling\\Assignment\\epa018aDeviceModelingAssignment\\IV Curves Data.txt"
file_path = './Data From ADS/IV Curves/IV Curves Data for Gm.txt'

# Load the data into a DataFrame, assuming tab-delimited data and a header row
df = pd.read_csv(file_path, sep='\t', header=0)

# Rename the columns to remove special characters, if necessary
df.columns = ['VGS', 'VDS', 'IDS']

# Convert the columns to numeric values if they are not already
df['VGS'] = pd.to_numeric(df['VGS'], errors='coerce')
df['VDS'] = pd.to_numeric(df['VDS'], errors='coerce')
df['IDS'] = pd.to_numeric(df['IDS'], errors='coerce')

# Filter out any rows that could not be converted to numbers
df = df.dropna()

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
