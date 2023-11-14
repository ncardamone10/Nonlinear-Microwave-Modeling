import pandas as pd

def read_and_combine_tables_v3(file_path):
    data_dict = {}
    current_header = None
    freq_column = 'freq'

    with open(file_path, 'r') as file:
        for line in file:
            # Check for a header line
            if line.startswith('freq'):
                current_header = line.strip().split('\t')[1]
            else:
                # Split the line into frequency and data
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    freq, data = parts
                    key = (freq, current_header)
                    data_dict[key] = data

    # Convert the dictionary to a list of tuples and then to a DataFrame
    data_list = [(key[0], key[1], value) for key, value in data_dict.items()]
    df = pd.DataFrame(data_list, columns=[freq_column, 'Header', 'Value'])
    df[freq_column] = df[freq_column].astype(float)  # Convert frequency to float
    df['Value'] = df['Value'].astype(float)  # Convert value to float

    # Pivot the table to get the desired format
    pivoted_df = df.pivot(index=freq_column, columns='Header', values='Value').reset_index()
    return pivoted_df

# Example usage
file_path = './Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V and paras.txt'  # Updated file path
combined_table_v3 = read_and_combine_tables_v3(file_path)

# Save the combined table to an Excel file
output_file = './Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V and paras.xlsx'  # Name of the output Excel file
combined_table_v3.to_excel(output_file, index=False)

print(f"Data exported successfully to {output_file}")
