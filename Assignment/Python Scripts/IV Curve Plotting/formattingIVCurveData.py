import pandas as pd

def convert_txt_to_excel(txt_file_path, excel_file_path):
    # Initialize a list to hold the data
    data_list = []

    # Read the file line by line
    with open(txt_file_path, 'r') as file:
        for line in file:
            # Skip lines that start with 'VGS' or are blank (headers or blank rows in the data)
            if line.startswith('VGS') or not line.strip():
                continue
            # Split the line on tabs and append to the data list
            data_list.append(line.split('\t'))

    # Convert the list to a DataFrame
    data = pd.DataFrame(data_list, columns=['VGS', 'VDS', 'IDS'])

    # Remove any rows that have all NaN or None values
    data.dropna(how='all', inplace=True)

    # Write the data into an Excel file
    data.to_excel(excel_file_path, index=False)

# Define the file paths
txt_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV Curves Updated2.txt'
excel_file_path = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/IV Curves/IV_Curves_Updated.xlsx'

# Call the function to convert the file
convert_txt_to_excel(txt_file_path, excel_file_path)
