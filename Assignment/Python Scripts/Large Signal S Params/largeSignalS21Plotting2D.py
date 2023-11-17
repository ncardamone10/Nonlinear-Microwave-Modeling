import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
excel_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/Large Signal S Params/largeSignalS21_VDS=2V_VGS=-0.7V.xlsx'
data = pd.read_excel(excel_file_path)

# Convert frequency to GHz
data['RFfreq'] = data['RFfreq'] / 1e9

# Calculate the magnitude of S21 in dB
data['magnitude_S21_dB'] = 20 * np.log10(np.sqrt(data['real(S(2,1))']**2 + data['imag(S(2,1))']**2))

# Define the specific input frequencies you're interested in
desired_input_frequencies = [1, 2, 5, 10, 12, 18, 30, 40]

# Define the specific power levels you're interested in
desired_power_levels = [-20, -10, -5, 0, 5, 10, 20]

# Create a 2x1 subplot
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Top subplot: Magnitude of S21 vs. Input Power
axs[0].set_title('EPA018A Large Signal |S21|, VDS=2V and VGS=-0.7V\nMagnitude of S21 vs Input Power')
axs[0].set_xlabel('Input Power (dB)')
axs[0].set_ylabel('Magnitude of S21 (dB)')

# Generate colors for the curves
color_list = plt.cm.rainbow(np.linspace(0, 1, len(desired_input_frequencies)))

# Temporary lists to store handles and labels for the legend
handles = []
labels = []

# Plot the magnitude of S21 curves for each input frequency or the closest one available
for desired_freq, color in zip(desired_input_frequencies, color_list):
    # Find the closest frequency values if the desired one is not available
    closest_freqs = data['RFfreq'].unique()
    closest_freq = min(closest_freqs, key=lambda x: abs(x - desired_freq))
    
    # Extract the data for the closest frequency
    data_for_freq = data[data['RFfreq'] == closest_freq]
    
    # Create the curve for magnitude of S21 vs. Input Power
    axs[0].plot(data_for_freq['RFpower'], data_for_freq['magnitude_S21_dB'], label=f'Freq = {closest_freq:.2f} GHz', color=color)
    
    # Append label to the list
    labels.append(f'Freq = {closest_freq:.2f} GHz')

# Add a legend to the top subplot
axs[0].legend(labels, title="Input Frequencies", loc="best")

# Bottom subplot: Magnitude of S21 vs. Frequency
axs[1].set_title('Magnitude of S21 vs Frequency')
axs[1].set_xlabel('Frequency (GHz)')
axs[1].set_ylabel('Magnitude of S21 (dB)')

# Generate colors for the curves
color_list = plt.cm.rainbow(np.linspace(0, 1, len(desired_power_levels)))

# Temporary lists to store handles and labels for the legend
handles = []
labels = []

# Plot the magnitude of S21 curves for each power level or the closest one available
for desired_power, color in zip(desired_power_levels, color_list):
    # Find the closest power values if the desired one is not available
    closest_powers = data['RFpower'].unique()
    closest_power = min(closest_powers, key=lambda x: abs(x - desired_power))
    
    # Extract the data for the closest power level
    data_for_power = data[data['RFpower'] == closest_power]
    
    # Create the curve for magnitude of S21 vs. Frequency
    axs[1].plot(data_for_power['RFfreq'], data_for_power['magnitude_S21_dB'], label=f'Power = {closest_power} dB', color=color)
    
    # Append label to the list
    labels.append(f'Power = {closest_power} dB')

# Add a legend to the bottom subplot
axs[1].legend(labels, title="Input Powers", loc="best")

# Add gridlines to both subplots
axs[0].grid(True)
axs[1].grid(True)

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()
