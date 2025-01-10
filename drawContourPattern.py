import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
import numpy as np
import math  # For rounding functions

# --------------------- Configuration ---------------------

# Define the frequency list (hardcoded at the beginning)
FREQUENCY_LIST = [1.52, 1.67]  # in GHz

# --------------------- Matplotlib Styles ---------------------

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

plt.rcParams['axes.prop_cycle'] = (
    cycler('color', ['#FF0000', '#FFAA00', '#58A500', '#00BFE9', '#2000AA', '#960096', '#808080'])
    + cycler('linestyle', ['-'] * 7)
    + cycler('linewidth', [2] * 7)
    + cycler('alpha', [1] * 7)
)

# --------------------- Helper Functions ---------------------

def is_valid_csv(filename):
    """
    Check if the file is a .csv file and does not start with 'sliced_'.
    """
    return filename.lower().endswith('.csv') and not os.path.basename(filename).startswith('sliced_')

def find_csv_files(directory):
    """
    Traverse the directory and subdirectories to find all valid CSV files.
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_valid_csv(file):
                csv_files.append(os.path.join(root, file))
    return csv_files

def sanitize_filename(name):
    """
    Sanitize the filename by replacing or removing invalid characters.
    """
    keep_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(c if c in keep_chars else '_' for c in name)

def complement_phi_theta(data):
    """
    Complement the phi to cover 0–360 degrees by mirroring theta and adding 180 to phi.
    Assumes original phi is from 0 to 180 degrees.
    """
    complementary = data.copy()
    complementary['Phi'] = (complementary['Phi'] + 180) % 360
    complementary['Theta'] = -complementary['Theta']
    return pd.concat([data, complementary], ignore_index=True)

# --------------------- Core Processing Routines ---------------------

def process_csv(file_path, frequency_list):
    """
    Process a single CSV file:
    - Extract data for specified frequencies and all phi/theta.
    - Complement Phi to cover 0–360 degrees.
    - Generate stepped contour plots for each relevant data column at each frequency.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    columns = data.columns.tolist()

    required_main_columns = {'Frequency', 'Phi', 'Theta'}
    if not required_main_columns.issubset(columns):
        print(f"Skipping {file_path}: Missing one of the required columns {required_main_columns}")
        return

    # Identify data columns (excluding 'Frequency', 'Phi', 'Theta')
    data_columns = [col for col in columns if col not in ['Frequency', 'Phi', 'Theta']]
    if not data_columns:
        print(f"No data columns to plot in {file_path}")
        return

    # Filter data for the specified frequencies
    data_filtered_freq = data[data['Frequency'].isin(frequency_list)]
    if data_filtered_freq.empty:
        print(f"No data matching the specified frequencies in {file_path}")
        return

    # Complement phi to cover 0–360 degrees
    data_complemented = complement_phi_theta(data_filtered_freq)
    # Ensure Phi is sorted for plotting
    data_complemented = data_complemented.sort_values(by=['Phi', 'Theta'])

    # Create the output directory
    csv_filename = os.path.basename(file_path)
    csv_name, _ = os.path.splitext(csv_filename)
    output_dir = os.path.join(os.path.dirname(file_path), csv_name)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Determine a global max across all non-axial-ratio, non-phase columns
    # so that they share the same range [global_max - 30, global_max].
    # -------------------------------------------------------------
    def is_phase_column(col_name):
        return ("Phase" in col_name)  # case-sensitive check
    
    normal_cols = [
        col for col in data_columns
        if col != 'Axial Ratio (dB)' and not is_phase_column(col)
    ]
    
    if normal_cols:
        global_max_value = max(data_complemented[ncol].max() for ncol in normal_cols)
        global_max_rounded = math.ceil(global_max_value / 5.0) * 5.0
        global_min_rounded = global_max_rounded - 30.0
        # Ensure they don't collapse (edge case)
        if global_min_rounded == global_max_rounded:
            global_max_rounded += 5.0
            global_min_rounded = global_max_rounded - 30.0
    else:
        # If there are no normal columns, we just set something
        global_max_rounded = 0.0
        global_min_rounded = -30.0

    # --------------------- Plot Each Column ---------------------
    for col in data_columns:
        data_col = data_complemented[['Frequency', 'Phi', 'Theta', col]].dropna()
        if data_col.empty:
            print(f"No data for column {col} in {file_path}")
            continue

        if col == 'Axial Ratio (dB)':
            # For axial ratio, we keep the 0–12 range in 1.5-dB stepped intervals
            Z_min_rounded = 0.0
            Z_max_rounded = 12.0
            step_size = 1.5
            levels = np.arange(Z_min_rounded, Z_max_rounded + step_size, step_size)
        elif "Phase" in col:
            # If it's a phase column, we directly use the data min and max
            Z_min_rounded = data_col[col].min()
            Z_max_rounded = data_col[col].max()
            # Choose the number of steps or resolution
            # (for example, 21 evenly spaced steps)
            num_steps = 21
            if Z_max_rounded > Z_min_rounded:  # to avoid zero division
                levels = np.linspace(Z_min_rounded, Z_max_rounded, num_steps)
            else:
                # Edge case: if no range is possible, define a dummy array:
                levels = np.array([Z_min_rounded, Z_min_rounded + 1.0])
        else:
            # Use the global min/max for all normal columns
            Z_min_rounded = global_min_rounded
            Z_max_rounded = global_max_rounded
            step = 5.0  # step size for discrete contour levels
            levels = np.arange(Z_min_rounded, Z_max_rounded + step, step)

        for freq in frequency_list:
            data_at_freq = data_col[data_col['Frequency'] == freq]
            if data_at_freq.empty:
                print(f"No data for column {col} at frequency {freq} GHz in {file_path}")
                continue

            # Identify unique phi/theta for meshgrid
            phi_unique = np.sort(data_at_freq['Phi'].unique())
            theta_unique = np.sort(data_at_freq['Theta'].unique())
            Phi, Theta = np.meshgrid(phi_unique, theta_unique)

            # Pivot data to create grid
            try:
                Z = data_at_freq.pivot_table(
                    index='Theta', columns='Phi', values=col
                ).reindex(index=theta_unique, columns=phi_unique)
            except Exception as e:
                print(f"Failed to pivot data for column {col} in {file_path}: {e}")
                continue

            # Interpolate missing data if needed
            if Z.isnull().values.any():
                print(f"Data contains NaN for column {col} at frequency {freq} GHz in {file_path}. Interpolating missing values.")
                Z = Z.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0)

            Z = Z.values

            # Create figure
            plt.figure(figsize=(10, 8))
            ax = plt.gca()

            # Generate stepped contour
            contour = ax.contourf(Phi, Theta, Z, levels=levels, cmap='jet', extend='both')

            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax, extend='both')
            cbar.set_label(col)

            # For colorbar ticks in normal columns, use multiples of 5
            if col == 'Axial Ratio (dB)':
                cbar.set_ticks(levels)
            elif "Phase" in col:
                # If it's a phase column, pick a few ticks in the [min, max] range
                cbar.set_ticks(np.linspace(Z_min_rounded, Z_max_rounded, 5))
            else:
                cbar_ticks = np.arange(Z_min_rounded, Z_max_rounded + step, step)
                cbar.set_ticks(cbar_ticks)

            # Labels and title
            ax.set_xlabel('Phi (degrees)')
            ax.set_ylabel('Theta (degrees)')
            ax.set_title(f'Stepped Contour Plot of {col} at {freq} GHz')

            # Show max/min values in data
            text_x = 0.95  # Relative (axes) position
            text_y = 0.95
            Z_max_data = np.nanmax(Z)
            Z_min_data = np.nanmin(Z)
            ax.text(
                text_x,
                text_y,
                f'Max: {Z_max_data:.2f}\nMin: {Z_min_data:.2f}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )

            plt.tight_layout()

            # Save figure
            safe_col_name = sanitize_filename(col)
            figure_filename = f'contour {freq}GHz {safe_col_name}.png'
            output_path = os.path.join(output_dir, figure_filename)
            try:
                plt.savefig(output_path, dpi=300)
                print(f"Saved contour plot to {output_path}")
            except Exception as e:
                print(f"Failed to save contour plot for {file_path}, column {col}, frequency {freq}: {e}")
            finally:
                plt.close()

# --------------------- Public Function for External Call ---------------------

def generate_contours(directory='.'):
    """
    High-level function for processing all valid CSV files in the given directory.
    Meant to be callable by external scripts.
    """
    base_directory = os.path.abspath(directory)
    csv_files = find_csv_files(base_directory)
    if not csv_files:
        print("No valid CSV files found.")
        return

    print(f"Found {len(csv_files)} valid CSV file(s). Processing...")

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        process_csv(csv_file, FREQUENCY_LIST)

    print("Processing completed.")

# --------------------- Main Function ---------------------

def main():
    """
    Entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Process CSV files and generate stepped contour plots.')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Path to the directory to scan. Defaults to current directory.'
    )
    args = parser.parse_args()

    # Call our high-level function
    generate_contours(args.directory)

# --------------------- Entry Point ---------------------

if __name__ == "__main__":
    main()