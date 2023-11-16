import pandas as pd
import pandas as pd
import os
import glob

def reformat_remove_nonnumeric_and_set_header(file_path, output_file_path):
    # Read the data file
    data = pd.read_csv(file_path, sep="\t", header=None)

    # Split the data into two halves
    mid_index = len(data) // 2
    first_half = data.iloc[:mid_index, :]
    second_half = data.iloc[mid_index:, :].reset_index(drop=True)

    # Concatenate the first three columns of the first half with the third column of the second half
    combined_data = pd.concat([first_half.iloc[:, :3], second_half.iloc[:, 2]], axis=1)

    # Set the first row as column names and remove any '[...]'
    combined_data.columns = combined_data.iloc[0].replace(r'\[.*?\]', '', regex=True)

    # Drop the first row now that it's been set as the header
    combined_data = combined_data[1:]

    # Function to check if a row contains only numeric values (excluding the header)
    def is_row_numeric(row):
        return row.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all()

    # Apply this function to filter out non-numeric rows, keeping the header
    combined_data = combined_data[combined_data.apply(is_row_numeric, axis=1) | (combined_data.index == 0)]

    # Export the cleaned data to Excel
    combined_data.to_excel(output_file_path, index=False)


# Directory containing the files
directory = './Nonlinear-Microwave-Modeling/Assignment/Data From ADS/Large Signal S Params/'

# Find all txt files starting with "largeSignalS" in the directory
txt_files = glob.glob(os.path.join(directory, 'largeSignalS*.txt'))

# Process each file
for file_path in txt_files:
    # Construct the output file path by replacing '.txt' with '.xlsx'
    output_file_path = file_path.replace('.txt', '.xlsx')

    # Process the data
    reformat_remove_nonnumeric_and_set_header(file_path, output_file_path)
    print(f"Processed and saved: {output_file_path}")

print("All files processed.")

