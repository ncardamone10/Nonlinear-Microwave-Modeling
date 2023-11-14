import pandas as pd

# Your file path
file_path = './Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.txt'

# Reading the data from the file
# The delimiter is a space, and we skip the first row which contains the column names
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)

# Adding column names
columns = ['frequencyGHz', 'S11_Magnitude', 'S11_Phase', 'S21_Magnitude', 'S21_Phase', 'S12_Magnitude', 'S12_Phase', 'S22_Magnitude', 'S22_Phase']
data.columns = columns

# Display the first few rows of your data
print(data.head())

# Save the data to an Excel file
excel_file = './Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx'
data.to_excel(excel_file, index=False)

print(f"Data saved to {excel_file}")
