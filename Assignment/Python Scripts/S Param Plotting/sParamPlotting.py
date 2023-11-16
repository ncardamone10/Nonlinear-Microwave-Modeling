# This just plots the data from ADS

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob
# import pandas as pd

# # Function to calculate the magnitude in dB of S parameters
# def calculate_magnitude_db(df):
#     magnitudes_db = {}
#     for param in ['1,1', '1,2', '2,1', '2,2']:
#         real_col = f'real(S({param}))'
#         imag_col = f'imag(S({param}))'
#         magnitude = np.sqrt(df[real_col]**2 + df[imag_col]**2)
#         magnitudes_db[param] = 20 * np.log10(magnitude)
#     return magnitudes_db

# # Function to plot the S parameters in a 2x2 grid
# def plot_s_parameters_combined(magnitudes_all_files, titles, params=['1,1', '1,2', '2,1', '2,2']):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle('S Parameters Magnitude in dB (Multiple Files)')

#     for i, param in enumerate(params):
#         ax = axs[i // 2, i % 2]
#         for file_index, magnitudes in enumerate(magnitudes_all_files):
#             ax.plot(magnitudes['freq'] / 1e9, magnitudes[param], label=titles[file_index])
#         ax.set_title(f'S{param}')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# # Path to the folder containing the Excel files
# folder_path = './Data From ADS/S Params'

# # Search for Excel files in the folder
# excel_files = glob.glob(os.path.join(folder_path, 'ADS S Params*.xlsx'))

# # Process each file and plot
# magnitudes_all_files = []
# titles = []
# for file in excel_files:
#     df = pd.read_excel(file)
#     magnitudes_db = calculate_magnitude_db(df)
#     magnitudes_db['freq'] = df['freq']  # Adding frequency for plotting
#     magnitudes_all_files.append(magnitudes_db)
#     titles.append(os.path.basename(file))

# # Plotting
# plot_s_parameters_combined(magnitudes_all_files, titles)

#----------------------------------------
# This plots the data from ADS and the datasheet, but all bias conditions are on the same plots
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob
# import pandas as pd

# # Function to calculate the magnitude in dB of S parameters
# def calculate_magnitude_db(df):
#     magnitudes_db = {}
#     for param in ['1,1', '1,2', '2,1', '2,2']:
#         real_col = f'real(S({param}))'
#         imag_col = f'imag(S({param}))'
#         magnitude = np.sqrt(df[real_col]**2 + df[imag_col]**2)
#         magnitudes_db[param] = 20 * np.log10(magnitude)
#     return magnitudes_db

# # Function to process the new Excel files format (Datasheet) and convert magnitudes to dB
# def process_new_file_format_db(file_path):
#     df = pd.read_excel(file_path)
#     magnitudes_db = {
#         '1,1': 20 * np.log10(df['S11_Magnitude']),
#         '2,1': 20 * np.log10(df['S21_Magnitude']),
#         '1,2': 20 * np.log10(df['S12_Magnitude']),
#         '2,2': 20 * np.log10(df['S22_Magnitude']),
#         'freq': df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
#     }
#     return magnitudes_db

# # Function to plot the S parameters in a 2x2 grid
# def plot_s_parameters_combined(magnitudes_all_files, titles, params=['1,1', '1,2', '2,1', '2,2']):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle('S Parameters Magnitude in dB (Multiple Files)')

#     for i, param in enumerate(params):
#         ax = axs[i // 2, i % 2]
#         for file_index, magnitudes in enumerate(magnitudes_all_files):
#             ax.plot(magnitudes['freq'] / 1e9, magnitudes[param], label=titles[file_index])  # Convert Hz back to GHz for plotting
#         ax.set_title(f'S{param}')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# # Path to the folder containing the Excel files
# folder_path = './Data From ADS/S Params'

# # Search for Excel files in the folder
# excel_files = glob.glob(os.path.join(folder_path, 'ADS S Params*.xlsx'))

# # Process each file and plot (ADS S Params)
# magnitudes_all_files_ads = []
# titles_ads = []
# for file in excel_files:
#     df = pd.read_excel(file)
#     magnitudes_db = calculate_magnitude_db(df)
#     magnitudes_db['freq'] = df['freq']  # Adding frequency for plotting
#     magnitudes_all_files_ads.append(magnitudes_db)
#     titles_ads.append(os.path.basename(file))

# # New file paths for the datasheet Excel files
# new_file_paths = [
#     './Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx',
#     './Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx'
# ]

# # Process each of the new files and plot (Datasheet)
# magnitudes_all_files_datasheet = []
# titles_datasheet = []
# for file_path in new_file_paths:
#     magnitudes_new_db = process_new_file_format_db(file_path)
#     magnitudes_all_files_datasheet.append(magnitudes_new_db)
#     titles_datasheet.append(os.path.basename(file_path))

# # Combine and plot both sets of files
# magnitudes_all_files_combined = magnitudes_all_files_ads + magnitudes_all_files_datasheet
# titles_combined = titles_ads + titles_datasheet
# plot_s_parameters_combined(magnitudes_all_files_combined, titles_combined)



# #----------------------------------
# # This splits off the plots based on bias conditions
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob
# import pandas as pd

# # Function to calculate the magnitude in dB of S parameters
# def calculate_magnitude_db(df):
#     magnitudes_db = {}
#     for param in ['1,1', '1,2', '2,1', '2,2']:
#         real_col = f'real(S({param}))'
#         imag_col = f'imag(S({param}))'
#         if real_col in df and imag_col in df:
#             magnitude = np.sqrt(df[real_col]**2 + df[imag_col]**2)
#             magnitudes_db[param] = 20 * np.log10(magnitude)
#         magnitudes_db['freq'] = df['freq']  # Assuming frequency is always present
#     return magnitudes_db

# # Function to process the new Excel files format (Datasheet) and convert magnitudes to dB
# def process_new_file_format_db(file_path):
#     df = pd.read_excel(file_path)
#     if 'frequencyGHz' in df.columns:
#         magnitudes_db = {
#             '1,1': 20 * np.log10(df['S11_Magnitude']),
#             '2,1': 20 * np.log10(df['S21_Magnitude']),
#             '1,2': 20 * np.log10(df['S12_Magnitude']),
#             '2,2': 20 * np.log10(df['S22_Magnitude']),
#             'freq': df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
#         }
#         return magnitudes_db
#     else:
#         return calculate_magnitude_db(df)

# # Function to plot the S parameters in a 2x2 grid
# def plot_s_parameters_combined(magnitudes_all_files, titles, plot_title, params=['1,1', '1,2', '2,1', '2,2']):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(plot_title)

#     for i, param in enumerate(params):
#         ax = axs[i // 2, i % 2]
#         for file_index, magnitudes in enumerate(magnitudes_all_files):
#             if 'freq' in magnitudes:
#                 ax.plot(magnitudes['freq'] / 1e9, magnitudes[param], label=titles[file_index])  # Convert Hz back to GHz for plotting
#             else:
#                 print(f"Frequency data missing in file: {titles[file_index]}")
#         ax.set_title(f'S{param}')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# # File paths for the different bias conditions
# file_paths_bias1 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=2V and VGS=-0.7V*.xlsx')
# file_paths_bias1.append('./Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx')

# file_paths_bias2 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V*.xlsx')
# file_paths_bias2.append('./Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx')

# # Process files for first bias condition
# magnitudes_bias1 = []
# titles_bias1 = []
# for file in file_paths_bias1:
#     magnitudes_bias1.append(process_new_file_format_db(file))
#     titles_bias1.append(os.path.basename(file))

# # Plot for first bias condition
# plot_s_parameters_combined(magnitudes_bias1, titles_bias1, "S Parameters for Bias Condition VDS=2V, VGS=-0.7V")

# # Process files for second bias condition
# magnitudes_bias2 = []
# titles_bias2 = []
# for file in file_paths_bias2:
#     magnitudes_bias2.append(process_new_file_format_db(file))
#     titles_bias2.append(os.path.basename(file))

# # Plot for second bias condition
# plot_s_parameters_combined(magnitudes_bias2, titles_bias2, "S Parameters for Bias Condition VDS=6V, VGS=-0.45V")


# #---------------------------------
# # This makes the legend look better
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob
# import pandas as pd

# # Function to calculate the magnitude in dB of S parameters
# def calculate_magnitude_db(df):
#     magnitudes_db = {}
#     for param in ['1,1', '1,2', '2,1', '2,2']:
#         real_col = f'real(S({param}))'
#         imag_col = f'imag(S({param}))'
#         if real_col in df and imag_col in df:
#             magnitude = np.sqrt(df[real_col]**2 + df[imag_col]**2)
#             magnitudes_db[param] = 20 * np.log10(magnitude)
#         magnitudes_db['freq'] = df['freq']
#     return magnitudes_db

# # Function to process the new Excel files format (Datasheet) and convert magnitudes to dB
# def process_new_file_format_db(file_path):
#     df = pd.read_excel(file_path)
#     if 'frequencyGHz' in df.columns:
#         magnitudes_db = {
#             '1,1': 20 * np.log10(df['S11_Magnitude']),
#             '2,1': 20 * np.log10(df['S21_Magnitude']),
#             '1,2': 20 * np.log10(df['S12_Magnitude']),
#             '2,2': 20 * np.log10(df['S22_Magnitude']),
#             'freq': df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
#         }
#         return magnitudes_db
#     else:
#         return calculate_magnitude_db(df)

# # Function to plot the S parameters in a 2x2 grid
# def plot_s_parameters_combined(magnitudes_all_files, titles, plot_title, params=['1,1', '1,2', '2,1', '2,2']):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(plot_title)

#     for i, param in enumerate(params):
#         ax = axs[i // 2, i % 2]
#         for file_index, magnitudes in enumerate(magnitudes_all_files):
#             if 'freq' in magnitudes:
#                 # Shorten the title for the legend
#                 legend_title = os.path.splitext(titles[file_index])[0].split(' with ')[-1]
#                 ax.plot(magnitudes['freq'] / 1e9, magnitudes[param], label=legend_title)
#             else:
#                 print(f"Frequency data missing in file: {titles[file_index]}")
#         ax.set_title(f'S{param}')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.grid(True)
#         ax.legend(loc='best', fontsize='small')

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# # File paths for the different bias conditions
# file_paths_bias1 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=2V and VGS=-0.7V*.xlsx')
# file_paths_bias1.append('./Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx')

# file_paths_bias2 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V*.xlsx')
# file_paths_bias2.append('./Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx')

# # Process files for first bias condition
# magnitudes_bias1 = []
# titles_bias1 = []
# for file in file_paths_bias1:
#     magnitudes_bias1.append(process_new_file_format_db(file))
#     titles_bias1.append(os.path.basename(file))

# # Plot for first bias condition
# plot_s_parameters_combined(magnitudes_bias1, titles_bias1, "S Parameters for Bias Condition VDS=2V, VGS=-0.7V")

# # Process files for second bias condition
# magnitudes_bias2 = []
# titles_bias2 = []
# for file in file_paths_bias2:
#     magnitudes_bias2.append(process_new_file_format_db(file))
#     titles_bias2.append(os.path.basename(file))

# # Plot for second bias condition
# plot_s_parameters_combined(magnitudes_bias2, titles_bias2, "S Parameters for Bias Condition VDS=6V, VGS=-0.45V")


# #--------------------------------
# # Adding in smith charts for S11 and S22
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import skrf as rf
# import glob

# # Function to process the Excel files and convert S parameters to complex numbers
# def process_file_to_complex(file_path):
#     df = pd.read_excel(file_path)
#     complex_s = {}
#     if 'frequencyGHz' in df.columns:
#         complex_s['freq'] = df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
#         for param in ['11', '12', '21', '22']:
#             magnitude = df[f'S{param}_Magnitude']
#             phase = df[f'S{param}_Phase'] * np.pi / 180  # Convert phase to radians
#             complex_s[param] = magnitude * np.exp(1j * phase)
#     else:
#         complex_s['freq'] = df['freq']
#         for param in ['1,1', '1,2', '2,1', '2,2']:
#             real_col = f'real(S({param}))'
#             imag_col = f'imag(S({param}))'
#             complex_s[param.replace(',', '')] = df[real_col] + 1j * df[imag_col]
#     return complex_s

# # Function to generate a descriptive legend title from the file name
# def generate_legend_title(file_name):
#     base_name = os.path.splitext(os.path.basename(file_name))[0]
#     if "Datasheet" in base_name:
#         return "Datasheet"
#     elif "and paras" in base_name:
#         return "ADS with Parasitics"
#     elif "no paras" in base_name:
#         return "ADS without Parasitics"
#     else:
#         return "Unknown Source"

# # Function to plot the 2x2 subplot matrix for each bias condition
# def plot_s_parameters_2x2(file_paths, title):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(title)

#     # Iterate over each file and plot on the respective subplot
#     for file_path in file_paths:
#         complex_s = process_file_to_complex(file_path)
#         legend_title = generate_legend_title(file_path)

#         # S11 on Smith chart
#         rf.plotting.plot_smith(complex_s['11'], ax=axs[0, 0], label=legend_title)
#         axs[0, 0].set_title('S11 Smith Chart')

#         # S12 magnitude plot
#         axs[0, 1].plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['12'])), label=legend_title)
#         axs[0, 1].set_title('S12 Magnitude (dB)')
#         axs[0, 1].set_xlabel('Frequency (GHz)')
#         axs[0, 1].set_ylabel('Magnitude (dB)')

#         # S21 magnitude plot
#         axs[1, 0].plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['21'])), label=legend_title)
#         axs[1, 0].set_title('S21 Magnitude (dB)')
#         axs[1, 0].set_xlabel('Frequency (GHz)')
#         axs[1, 0].set_ylabel('Magnitude (dB)')

#         # S22 on Smith chart
#         rf.plotting.plot_smith(complex_s['22'], ax=axs[1, 1], label=legend_title)
#         axs[1, 1].set_title('S22 Smith Chart')

#     # Adding legends and grid
#     for ax in axs.flat:
#         ax.legend(loc='best', fontsize='small')
#         ax.grid(True)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# # File paths for the different bias conditions
# file_paths_bias1 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=2V and VGS=-0.7V*.xlsx')
# file_paths_bias1.append('./Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx')

# file_paths_bias2 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V*.xlsx')
# file_paths_bias2.append('./Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx')

# # Plot 2x2 matrix for first bias condition
# plot_s_parameters_2x2(file_paths_bias1, "S Parameters for Bias Condition VDS=2V, VGS=-0.7V")

# # Plot 2x2 matrix for second bias condition
# plot_s_parameters_2x2(file_paths_bias2, "S Parameters for Bias Condition VDS=6V, VGS=-0.45V")

# #-------------------------------------
# # This removes the axis labels on the smith chart and scales the S12 plot correctly
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import skrf as rf
# import glob

# # Function to process the Excel files and convert S parameters to complex numbers
# def process_file_to_complex(file_path):
#     df = pd.read_excel(file_path)
#     complex_s = {}
#     if 'frequencyGHz' in df.columns:
#         complex_s['freq'] = df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
#         for param in ['11', '12', '21', '22']:
#             magnitude = df[f'S{param}_Magnitude']
#             phase = df[f'S{param}_Phase'] * np.pi / 180  # Convert phase to radians
#             complex_s[param] = magnitude * np.exp(1j * phase)
#     else:
#         complex_s['freq'] = df['freq']
#         for param in ['1,1', '1,2', '2,1', '2,2']:
#             real_col = f'real(S({param}))'
#             imag_col = f'imag(S({param}))'
#             complex_s[param.replace(',', '')] = df[real_col] + 1j * df[imag_col]
#     return complex_s

# # Function to generate a descriptive legend title from the file name
# def generate_legend_title(file_name):
#     base_name = os.path.splitext(os.path.basename(file_name))[0]
#     if "Datasheet" in base_name:
#         return "Datasheet"
#     elif "and paras" in base_name:
#         return "ADS with Parasitics"
#     elif "no paras" in base_name:
#         return "ADS without Parasitics"
#     else:
#         return "Unknown Source"

# # Function to calculate reflection coefficient for given r and x
# def calculate_reflection_coefficient(r, x):
#     z = r + 1j * x  # Normalized impedance
#     gamma = (z - 1) / (z + 1)  # Reflection coefficient
#     return gamma

# # Function to annotate specific points on the Smith chart
# def annotate_smith_points(ax):
#     # List of points to annotate (normalized resistance r, normalized reactance x)
#     points = [(1, 0), (1, 1), (1, -1), (0.5, 0), (2, 0), (1, 2), (1, -2)]
#     for r, x in points:
#         gamma = calculate_reflection_coefficient(r, x)
#         ax.plot(np.real(gamma), np.imag(gamma), 'x', color='black')  # Plot the point as a small black "x"
#         label = f'r={r}' if x == 0 else f'x={x}'
#         ax.text(np.real(gamma), np.imag(gamma), label, ha='center', va='center', fontsize=8, color='black')

# # Function to plot the 2x2 subplot matrix for each bias condition
# def plot_s_parameters_2x2(file_paths, title):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(title)

#     # Iterate over each file and plot on the respective subplot
#     for file_path in file_paths:
#         complex_s = process_file_to_complex(file_path)
#         legend_title = generate_legend_title(file_path)

#         # S11 on Smith chart
#         ax = axs[0, 0]
#         rf.plotting.plot_smith(complex_s['11'], ax=ax, label=legend_title)
#         ax.set_title('S11 Smith Chart')
#         annotate_smith_points(ax)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize='small')

#         # S12 magnitude plot
#         ax = axs[0, 1]
#         ax.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['12'])), label=legend_title)
#         ax.set_title('S12 Magnitude (dB)')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.set_ylim(-50, 0)
#         ax.legend(loc='best', fontsize='small')
#         ax.grid(True)

#         # S21 magnitude plot
#         ax = axs[1, 0]
#         ax.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['21'])), label=legend_title)
#         ax.set_title('S21 Magnitude (dB)')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.legend(loc='best', fontsize='small')
#         ax.grid(True)

#         # S22 on Smith chart
#         ax = axs[1, 1]
#         rf.plotting.plot_smith(complex_s['22'], ax=ax, label=legend_title)
#         ax.set_title('S22 Smith Chart')
#         annotate_smith_points(ax)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize='small')

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# # File paths for the different bias conditions
# file_paths_bias1 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=2V and VGS=-0.7V*.xlsx')
# file_paths_bias1.append('./Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx')

# file_paths_bias2 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V*.xlsx')
# file_paths_bias2.append('./Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx')

# # Plot 2x2 matrix for first bias condition
# plot_s_parameters_2x2(file_paths_bias1, "S Parameters for Bias Condition VDS=2V, VGS=-0.7V")

# # Plot 2x2 matrix for second bias condition
# plot_s_parameters_2x2(file_paths_bias2, "S Parameters for Bias Condition VDS=6V, VGS=-0.45V")

#------------------------------------------------------------
# This removes the black x from the points on the smith chart, and changes various colours (chat gpt 3.5)
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import skrf as rf
# import glob

# # Function to process the Excel files and convert S parameters to complex numbers
# def process_file_to_complex(file_path):
#     df = pd.read_excel(file_path)
#     complex_s = {}
#     if 'frequencyGHz' in df.columns:
#         complex_s['freq'] = df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
#         for param in ['11', '12', '21', '22']:
#             magnitude = df[f'S{param}_Magnitude']
#             phase = df[f'S{param}_Phase'] * np.pi / 180  # Convert phase to radians
#             complex_s[param] = magnitude * np.exp(1j * phase)
#     else:
#         complex_s['freq'] = df['freq']
#         for param in ['1,1', '1,2', '2,1', '2,2']:
#             real_col = f'real(S({param}))'
#             imag_col = f'imag(S({param}))'
#             complex_s[param.replace(',', '')] = df[real_col] + 1j * df[imag_col]
#     return complex_s

# # Function to generate a descriptive legend title from the file name
# def generate_legend_title(file_name):
#     base_name = os.path.splitext(os.path.basename(file_name))[0]
#     if "Datasheet" in base_name:
#         return "Datasheet"
#     elif "and paras" in base_name:
#         return "ADS with Parasitics"
#     elif "no paras" in base_name:
#         return "ADS without Parasitics"
#     else:
#         return "Unknown Source"

# # Function to calculate reflection coefficient for given r and x
# def calculate_reflection_coefficient(r, x):
#     z = r + 1j * x  # Normalized impedance
#     gamma = (z - 1) / (z + 1)  # Reflection coefficient
#     return gamma

# # Function to annotate specific points on the Smith chart
# def annotate_smith_points(ax):
#     # List of points to annotate (normalized resistance r, normalized reactance x)
#     points = [(1, 0), (1, 1), (1, -1), (0.5, 0), (2, 0), (1, 2), (1, -2)]
#     for r, x in points:
#         gamma = calculate_reflection_coefficient(r, x)
#         # Plot the point as a small black "x" with zero alpha (invisible)
#         ax.plot(np.real(gamma), np.imag(gamma), 'x', color='black', alpha=0)
#         label = f'r={r}' if x == 0 else f'x={x}'
#         ax.text(np.real(gamma), np.imag(gamma), label, ha='center', va='center', fontsize=11, color='limegreen', weight='bold')

# # Function to plot the 2x2 subplot matrix for each bias condition
# def plot_s_parameters_2x2(file_paths, title):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(title)

#     # Define a list of colors for the curves
#     curve_colors = ['red', 'grey', 'blue', 'green']  # Add more colors if needed

#     # Iterate over each file and plot on the respective subplot
#     for i, file_path in enumerate(file_paths):
#         complex_s = process_file_to_complex(file_path)
#         legend_title = generate_legend_title(file_path)
#         curve_color = curve_colors[i]

#         # S11 on Smith chart
#         ax = axs[0, 0]
        
#         ax.axis('off')  # Turn off axes
#         annotate_smith_points(ax)
#         rf.plotting.plot_smith(complex_s['11'], ax=ax, label=legend_title, color=curve_color)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize='small')
#         ax.set_title('S11')

#         # S12 magnitude plot
#         ax = axs[0, 1]
#         ax.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['12'])), label=legend_title, color=curve_color)
#         ax.set_title('S12 Magnitude (dB)')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.set_ylim(-50, 0)
#         ax.legend(loc='best', fontsize='small')
#         ax.grid(True)

#         # S21 magnitude plot
#         ax = axs[1, 0]
#         ax.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['21'])), label=legend_title, color=curve_color)
#         ax.set_title('S21 Magnitude (dB)')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.legend(loc='best', fontsize='small')
#         ax.grid(True)

#         # S22 on Smith chart
#         ax = axs[1, 1]
        
#         ax.axis('off')  # Turn off axes
#         annotate_smith_points(ax)
#         rf.plotting.plot_smith(complex_s['22'], ax=ax, label=legend_title, color=curve_color)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize='small')
#         ax.set_title('S22')

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# # File paths for the different bias conditions
# file_paths_bias1 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=2V and VGS=-0.7V*.xlsx')
# file_paths_bias1.append('./Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx')

# file_paths_bias2 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V*.xlsx')
# file_paths_bias2.append('./Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx')

# # Plot 2x2 matrix for first bias condition
# plot_s_parameters_2x2(file_paths_bias1, "S Parameters for Bias Condition VDS=2V, VGS=-0.7V")

# # Plot 2x2 matrix for second bias condition
# plot_s_parameters_2x2(file_paths_bias2, "S Parameters for Bias Condition VDS=6V, VGS=-0.45V")


# #---------------------------------------------------
# # Adding the phase of S21 and S12 (back to chat gpt 4)
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import skrf as rf
# import glob

# # Function to process the Excel files and convert S parameters to complex numbers
# def process_file_to_complex(file_path):
#     df = pd.read_excel(file_path)
#     complex_s = {}
#     if 'frequencyGHz' in df.columns:
#         complex_s['freq'] = df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
#         for param in ['11', '12', '21', '22']:
#             magnitude = df[f'S{param}_Magnitude']
#             phase = df[f'S{param}_Phase'] * np.pi / 180  # Convert phase to radians
#             complex_s[param] = magnitude * np.exp(1j * phase)
#     else:
#         complex_s['freq'] = df['freq']
#         for param in ['1,1', '1,2', '2,1', '2,2']:
#             real_col = f'real(S({param}))'
#             imag_col = f'imag(S({param}))'
#             complex_s[param.replace(',', '')] = df[real_col] + 1j * df[imag_col]
#     return complex_s

# # Function to generate a descriptive legend title from the file name
# def generate_legend_title(file_name):
#     base_name = os.path.splitext(os.path.basename(file_name))[0]
#     if "Datasheet" in base_name:
#         return "Datasheet"
#     elif "and paras" in base_name:
#         return "ADS with Parasitics"
#     elif "no paras" in base_name:
#         return "ADS without Parasitics"
#     else:
#         return "Unknown Source"

# # Function to calculate reflection coefficient for given r and x
# def calculate_reflection_coefficient(r, x):
#     z = r + 1j * x  # Normalized impedance
#     gamma = (z - 1) / (z + 1)  # Reflection coefficient
#     return gamma

# # Function to annotate specific points on the Smith chart
# def annotate_smith_points(ax):
#     # List of points to annotate (normalized resistance r, normalized reactance x)
#     points = [(1, 0), (1, 1), (1, -1), (0.5, 0), (2, 0), (1, 2), (1, -2)]
#     for r, x in points:
#         gamma = calculate_reflection_coefficient(r, x)
#         # Plot the point as a small black "x" with zero alpha (invisible)
#         ax.plot(np.real(gamma), np.imag(gamma), 'x', color='black', alpha=0)
#         label = f'r={r}' if x == 0 else f'x={x}'
#         ax.text(np.real(gamma), np.imag(gamma), label, ha='center', va='center', fontsize=11, color='limegreen', weight='bold')

# # Function to plot the 2x2 subplot matrix for each bias condition
# def plot_s_parameters_2x2(file_paths, title):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(title)

#     curve_colors = ['red', 'grey', 'blue', 'green']

#     # Create variables to hold the secondary axis for phase plots
#     ax_phase_s21 = None
#     ax_phase_s12 = None

#     for i, file_path in enumerate(file_paths):
#         complex_s = process_file_to_complex(file_path)
#         legend_title = generate_legend_title(file_path)
#         curve_color = curve_colors[i]

#         # S11 on Smith chart
#         ax = axs[0, 0]
#         ax.axis('off')
#         annotate_smith_points(ax)
#         rf.plotting.plot_smith(complex_s['11'], ax=ax, label=legend_title, color=curve_color)
#         ax.set_title('S11')

#         # S12 magnitude and phase plot
#         ax = axs[0, 1]
#         # Plot magnitude
#         ax.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['12'])), label=f'{legend_title} Magnitude', color=curve_color)
#         ax.set_title('S12 Magnitude (dB) and Phase (degrees)')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.grid(True)

#         # Setup secondary axis for phase only once
#         if ax_phase_s12 is None:
#             ax_phase_s12 = ax.twinx()
#             ax_phase_s12.set_ylabel('Phase (degrees)')

#         # Plot phase
#         phase_degrees_s12 = np.angle(complex_s['12'], deg=True)
#         ax_phase_s12.plot(complex_s['freq'] / 1e9, phase_degrees_s12, linestyle='--', label=f'{legend_title} Phase', color=curve_color)

#         # S21 magnitude and phase plot
#         ax = axs[1, 0]
#         # Plot magnitude
#         ax.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['21'])), label=f'{legend_title} Magnitude', color=curve_color)
#         ax.set_title('S21 Magnitude (dB) and Phase (degrees)')
#         ax.set_xlabel('Frequency (GHz)')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.grid(True)

#         # Setup secondary axis for phase only once
#         if ax_phase_s21 is None:
#             ax_phase_s21 = ax.twinx()
#             ax_phase_s21.set_ylabel('Phase (degrees)')

#         # Plot phase
#         phase_degrees_s21 = np.angle(complex_s['21'], deg=True)
#         ax_phase_s21.plot(complex_s['freq'] / 1e9, phase_degrees_s21, linestyle='--', label=f'{legend_title} Phase', color=curve_color)

#         # S22 on Smith chart
#         ax = axs[1, 1]
#         ax.axis('off')
#         annotate_smith_points(ax)
#         rf.plotting.plot_smith(complex_s['22'], ax=ax, label=legend_title, color=curve_color)
#         ax.set_title('S22')

#     # Create combined legends for S12 and S21 subplots
#     for ax_main, ax_phase in [(axs[0, 1], ax_phase_s12), (axs[1, 0], ax_phase_s21)]:
#         handles, labels = [], []
#         for ax in [ax_main, ax_phase]:
#             h, l = ax.get_legend_handles_labels()
#             handles.extend(h)
#             labels.extend(l)
#         ax_main.legend(handles, labels, loc='upper left', fontsize='small')

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()


# # File paths for the different bias conditions
# file_paths_bias1 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=2V and VGS=-0.7V*.xlsx')
# file_paths_bias1.append('./Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx')

# file_paths_bias2 = glob.glob('./Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V*.xlsx')
# file_paths_bias2.append('./Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx')

# # Plot 2x2 matrix for first bias condition
# plot_s_parameters_2x2(file_paths_bias1, "S Parameters for Bias Condition VDS=2V, VGS=-0.7V")

# # Plot 2x2 matrix for second bias condition
# plot_s_parameters_2x2(file_paths_bias2, "S Parameters for Bias Condition VDS=6V, VGS=-0.45V")

#---------------------------------------------------
# Splitting the mag and phase of S12 and S21
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
import glob

# Function to process the Excel files and convert S parameters to complex numbers
def process_file_to_complex(file_path):
    df = pd.read_excel(file_path)
    complex_s = {}
    if 'frequencyGHz' in df.columns:
        complex_s['freq'] = df['frequencyGHz'] * 1e9  # Convert frequency from GHz to Hz
        for param in ['11', '12', '21', '22']:
            magnitude = df[f'S{param}_Magnitude']
            phase = df[f'S{param}_Phase'] * np.pi / 180  # Convert phase to radians
            complex_s[param] = magnitude * np.exp(1j * phase)
    else:
        complex_s['freq'] = df['freq']
        for param in ['1,1', '1,2', '2,1', '2,2']:
            real_col = f'real(S({param}))'
            imag_col = f'imag(S({param}))'
            complex_s[param.replace(',', '')] = df[real_col] + 1j * df[imag_col]
    return complex_s

# Function to generate a descriptive legend title from the file name
def generate_legend_title(file_name):
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    if "Datasheet" in base_name:
        return "Datasheet"
    elif "and paras" in base_name:
        return "ADS with Parasitics"
    elif "no paras" in base_name:
        return "ADS without Parasitics"
    else:
        return "Unknown Source"

# Function to calculate reflection coefficient for given r and x
def calculate_reflection_coefficient(r, x):
    z = r + 1j * x  # Normalized impedance
    gamma = (z - 1) / (z + 1)  # Reflection coefficient
    return gamma

# Function to annotate specific points on the Smith chart
def annotate_smith_points(ax):
    # List of points to annotate (normalized resistance r, normalized reactance x)
    points = [(1, 0), (1, 1), (1, -1), (0.5, 0), (2, 0), (1, 2), (1, -2)]
    for r, x in points:
        gamma = calculate_reflection_coefficient(r, x)
        # Plot the point as a small black "x" with zero alpha (invisible)
        ax.plot(np.real(gamma), np.imag(gamma), 'x', color='black', alpha=0)
        label = f'r={r}' if x == 0 else f'x={x}'
        ax.text(np.real(gamma), np.imag(gamma), label, ha='center', va='center', fontsize=11, color='limegreen', weight='bold')

# Function to plot the 2x2 subplot matrix for each bias condition
def plot_s_parameters_2x2(file_paths, title):
    fig = plt.figure(figsize=(14, 16))  # Adjusted figure size
    fig.suptitle(title)

    curve_colors = ['red', 'grey', 'blue', 'green']

    gs = fig.add_gridspec(4, 2)  # Create a 4x2 grid

    # Create subplots for Smith charts (S11 and S22) with double height
    ax_s11 = fig.add_subplot(gs[0:2, 0])
    ax_s22 = fig.add_subplot(gs[2:4, 1])

    # Initialize subplots for S12 and S21 magnitude and phase
    ax_s12_mag = fig.add_subplot(gs[0, 1])
    ax_s12_phase = fig.add_subplot(gs[1, 1], sharex=ax_s12_mag)
    ax_s21_mag = fig.add_subplot(gs[2, 0])
    ax_s21_phase = fig.add_subplot(gs[3, 0], sharex=ax_s21_mag)

    for i, file_path in enumerate(file_paths):
        complex_s = process_file_to_complex(file_path)
        legend_title = generate_legend_title(file_path)
        curve_color = curve_colors[i]

        # S11 on Smith chart
        ax_s11.axis('off')
        annotate_smith_points(ax_s11)
        rf.plotting.plot_smith(complex_s['11'], ax=ax_s11, label=legend_title, color=curve_color)

        # S12 magnitude and phase plots
        ax_s12_mag.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['12'])), label=f'{legend_title} Magnitude', color=curve_color)
        phase_degrees_s12 = np.angle(complex_s['12'], deg=True)
        ax_s12_phase.plot(complex_s['freq'] / 1e9, phase_degrees_s12, linestyle='--', label=f'{legend_title} Phase', color=curve_color)

        # S21 magnitude and phase plots
        ax_s21_mag.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['21'])), label=f'{legend_title} Magnitude', color=curve_color)
        phase_degrees_s21 = np.angle(complex_s['21'], deg=True)
        ax_s21_phase.plot(complex_s['freq'] / 1e9, phase_degrees_s21, linestyle='--', label=f'{legend_title} Phase', color=curve_color)

        # S22 on Smith chart
        ax_s22.axis('off')
        annotate_smith_points(ax_s22)
        rf.plotting.plot_smith(complex_s['22'], ax=ax_s22, label=legend_title, color=curve_color)

    # Set titles and labels for the subplots
    ax_s11.set_title('S11')
    ax_s12_mag.set_title('S12 Magnitude (dB)')
    ax_s12_phase.set_title('S12 Phase (degrees)')
    ax_s21_mag.set_title('S21 Magnitude (dB)')
    ax_s21_phase.set_title('S21 Phase (degrees)')
    ax_s22.set_title('S22')

    ax_s12_mag.set_xlabel('Frequency (GHz)')
    ax_s12_mag.set_ylabel('Magnitude (dB)')
    ax_s12_phase.set_xlabel('Frequency (GHz)')
    ax_s12_phase.set_ylabel('Phase (degrees)')
    ax_s21_mag.set_xlabel('Frequency (GHz)')
    ax_s21_mag.set_ylabel('Magnitude (dB)')
    ax_s21_phase.set_xlabel('Frequency (GHz)')
    ax_s21_phase.set_ylabel('Phase (degrees)')

    # Set y-axis limits and ticks for the phase plots
    phase_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    ax_s12_phase.set_ylim(-180, 180)
    ax_s12_phase.set_yticks(phase_ticks)
    ax_s21_phase.set_ylim(-180, 180)
    ax_s21_phase.set_yticks(phase_ticks)

    # Set y-axis limits and ticks for the S12 and S21 magnitude plots
    ax_s12_mag.set_ylim(-40, 5)
    ax_s12_mag.set_yticks(range(-40, 6, 5))
    ax_s21_mag.set_ylim(-6, 18)
    ax_s21_mag.set_yticks(range(-6, 19, 3))

    # Adjust legends for S12 and S21 to the right of the plot
    ax_s12_mag.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_s12_phase.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_s21_mag.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_s21_phase.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    # Adjust legends for S11 and S22 to the right of the plot
    ax_s11.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_s22.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    ax_s12_mag.grid(True)
    ax_s12_phase.grid(True)
    ax_s21_mag.grid(True)
    ax_s21_phase.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.5, right=0.8)

    plt.show()


# File paths for the different bias conditions
file_paths_bias1 = glob.glob('./Nonlinear-Microwave-Modeling/Assignment/Data From ADS/S Params/ADS S Params with VDS=2V and VGS=-0.7V*.xlsx')
file_paths_bias1.append('./Nonlinear-Microwave-Modeling/Assignment/Data From ADS/S Params/S Params From Datasheet VDS=2V and VGS=-0.7V.xlsx')

file_paths_bias2 = glob.glob('./Nonlinear-Microwave-Modeling/Assignment/Data From ADS/S Params/ADS S Params with VDS=6V and VGS=-0.45V*.xlsx')
file_paths_bias2.append('./Nonlinear-Microwave-Modeling/Assignment/Data From ADS/S Params/S Params From Datasheet VDS=6V and VGS=-0.45V.xlsx')

# Plot 2x2 matrix for first bias condition
plot_s_parameters_2x2(file_paths_bias1, "S Parameters for Bias Condition VDS=2V, VGS=-0.7V")

# Plot 2x2 matrix for second bias condition
plot_s_parameters_2x2(file_paths_bias2, "S Parameters for Bias Condition VDS=6V, VGS=-0.45V")








