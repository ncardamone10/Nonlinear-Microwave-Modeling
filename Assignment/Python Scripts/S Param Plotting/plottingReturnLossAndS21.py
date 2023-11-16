import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
import glob

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

def annotate_smith_points(ax):
    # List of points to annotate (normalized resistance r, normalized reactance x)
    points = [(1, 0), (1, 1), (1, -1), (0.5, 0), (2, 0), (1, 2), (1, -2)]
    for r, x in points:
        gamma = calculate_reflection_coefficient(r, x)
        # Plot the point as a small black "x" with zero alpha (invisible)
        ax.plot(np.real(gamma), np.imag(gamma), 'x', color='black', alpha=0)
        label = f'r={r}' if x == 0 else f'x={x}'
        ax.text(np.real(gamma), np.imag(gamma), label, ha='center', va='center', fontsize=11, color='limegreen', weight='bold')

def calculate_reflection_coefficient(r, x):
    z = r + 1j * x  # Normalized impedance
    gamma = (z - 1) / (z + 1)  # Reflection coefficient
    return gamma

def plot_return_loss_and_s21(file_paths, title):
    fig = plt.figure(figsize=(14, 16))  # Adjusted figure size
    fig.suptitle(title)

    curve_colors = ['red', 'grey', 'blue', 'green']

    gs = fig.add_gridspec(2, 2)  # Create a 2x2 grid

    ax_return_loss = fig.add_subplot(gs[0, 0])
    ax_s21_mag = fig.add_subplot(gs[0, 1])
    ax_phase_return_loss = fig.add_subplot(gs[1, 0])
    ax_s21_phase = fig.add_subplot(gs[1, 1])

    for i, file_path in enumerate(file_paths):
        complex_s = process_file_to_complex(file_path)
        legend_title = generate_legend_title(file_path)
        curve_color = curve_colors[i]

        # Calculate return loss (S11 magnitude in dB)
        return_loss = -20 * np.log10(np.abs(complex_s['11']))
        ax_return_loss.plot(complex_s['freq'] / 1e9, return_loss, label=f'{legend_title}', color=curve_color)

        # Calculate phase of return loss
        phase_degrees_return_loss = np.angle(complex_s['11'], deg=True)
        ax_phase_return_loss.plot(complex_s['freq'] / 1e9, phase_degrees_return_loss, linestyle='--', label=f'{legend_title}', color=curve_color)

        # Plot S21 magnitude in dB
        ax_s21_mag.plot(complex_s['freq'] / 1e9, 20 * np.log10(np.abs(complex_s['21'])), label=f'{legend_title}', color=curve_color)

        # Plot phase of S21
        phase_degrees_s21 = np.angle(complex_s['21'], deg=True)
        ax_s21_phase.plot(complex_s['freq'] / 1e9, phase_degrees_s21, linestyle='--', label=f'{legend_title}', color=curve_color)

    # Set titles and labels for the subplots
    ax_return_loss.set_title('Return Loss (dB)')
    ax_s21_mag.set_title('S21 Magnitude (dB)')
    ax_phase_return_loss.set_title('Phase of Return Loss (degrees)')
    ax_s21_phase.set_title('Phase of S21 (degrees)')

    ax_return_loss.set_xlabel('Frequency (GHz)')
    ax_return_loss.set_ylabel('Return Loss (dB)')
    ax_s21_mag.set_xlabel('Frequency (GHz)')
    ax_s21_mag.set_ylabel('S21 Magnitude (dB)')
    ax_phase_return_loss.set_xlabel('Frequency (GHz)')
    ax_phase_return_loss.set_ylabel('Phase (degrees)')
    ax_s21_phase.set_xlabel('Frequency (GHz)')
    ax_s21_phase.set_ylabel('Phase (degrees)')

    # Set y-axis limits and ticks for the phase plots
    phase_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    ax_phase_return_loss.set_ylim(-180, 180)
    ax_phase_return_loss.set_yticks(phase_ticks)
    ax_s21_phase.set_ylim(-180, 180)
    ax_s21_phase.set_yticks(phase_ticks)

    # Adjust legends to the right of the plot
    ax_return_loss.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_s21_mag.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_phase_return_loss.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_s21_phase.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    ax_return_loss.grid(True)
    ax_s21_mag.grid(True)
    ax_phase_return_loss.grid(True)
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

# Plot for first bias condition
plot_return_loss_and_s21(file_paths_bias1, "Return Loss and S21 for Bias Condition VDS=2V, VGS=-0.7V")

# Plot for second bias condition
plot_return_loss_and_s21(file_paths_bias2, "Return Loss and S21 for Bias Condition VDS=6V, VGS=-0.45V")
