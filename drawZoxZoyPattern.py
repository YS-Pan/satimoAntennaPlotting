import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
import numpy as np
import math

# --------------------- Configuration ---------------------
REQUIRED_GAIN_GROUPS = [
    ["Gain . dB", "Gain Phi. dB", "Gain Theta. dB"],
    ["Polar LC . Amp dB", "Polar RC . Amp dB"]
]

# Define the default frequency list (modify as needed)
FREQUENCY_LIST = [1.52, 1.67]

# Define the default phi list (modify as needed)
PHI_LIST = [0, 90]

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

def is_valid_csv(filename: str) -> bool:
    """
    Check if the file is a .csv file and does not start with 'sliced_'.
    """
    return filename.lower().endswith('.csv') and not os.path.basename(filename).startswith('sliced_')

def find_csv_files(directory: str) -> list:
    """
    Traverse the directory and subdirectories to find all valid CSV files.
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_valid_csv(file):
                csv_files.append(os.path.join(root, file))
    return csv_files

def sanitize_filename(name: str) -> str:
    """
    Sanitize the filename by replacing or removing invalid characters.
    """
    keep_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(c if c in keep_chars else '_' for c in name)

def get_r_limits(gain_values, range_diff=30):
    """
    Calculate r_max as the smallest multiple of 5 >= max_gain.
    Set r_min as (r_max - range_diff).
    """
    if not gain_values:
        return None, None
    max_gain = max(gain_values)
    r_max = math.ceil(max_gain / 5) * 5
    r_min = r_max - range_diff
    return r_max, r_min

def process_csv(file_path, frequency_list, phi_list):
    """
    Process a single CSV file:
      1. Check for required columns (Frequency, Phi, Theta).
      2. Filter data according to frequency_list and phi_list.
      3. Determine the global r_min and r_max from all sublists in REQUIRED_GAIN_GROUPS.
      4. For each freq and phi, plot each sublist separately, using the global r_min, r_max.
      5. Save figures in a folder with the same name as the CSV.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    columns = data.columns.tolist()
    required_main_columns = ['Frequency', 'Phi', 'Theta']
    if not all(col in columns for col in required_main_columns):
        print(f"Skipping {file_path}: Missing one of the required columns {required_main_columns}")
        return

    data_filtered = data[
        data['Frequency'].isin(frequency_list) &
        data['Phi'].isin(phi_list)
    ]
    if data_filtered.empty:
        print(f"No data matching the specified frequencies and phis in {file_path}")
        return

    # ---------------------------------------------------------
    # Compute a global r_min, r_max for all sublists in REQUIRED_GAIN_GROUPS
    # ---------------------------------------------------------
    # Collect all columns from all sublists that actually exist in this CSV.
    all_gain_columns = set()
    for sublist in REQUIRED_GAIN_GROUPS:
        for col in sublist:
            if col in columns:
                all_gain_columns.add(col)

    # Flatten all gain values from data_filtered for these columns
    all_gains = []
    if all_gain_columns:
        for col in all_gain_columns:
            col_values = data_filtered[col].dropna().values
            all_gains.extend(col_values)

    global_r_max, global_r_min = get_r_limits(all_gains, range_diff=30)
    if global_r_max is None or global_r_min is None:
        print(f"No valid gain data to compute r-limits for {file_path}")
        return

    print(f"Processing {file_path} with r-axis limits: {global_r_min} dB to {global_r_max} dB")

    # ---------------------------------------------------------
    # Create output directory (same name as CSV file)
    # ---------------------------------------------------------
    csv_filename = os.path.basename(file_path)
    csv_name, _ = os.path.splitext(csv_filename)
    output_dir = os.path.join(os.path.dirname(file_path), csv_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Now plot for each freq, phi, and each sublist in REQUIRED_GAIN_GROUPS
    # with the same r_min, r_max for each plot
    # ---------------------------------------------------------
    for freq in frequency_list:
        freq_data = data_filtered[data_filtered['Frequency'] == freq]
        if freq_data.empty:
            print(f"No data for frequency {freq} GHz in {file_path}")
            continue

        for phi in phi_list:
            phi_data = freq_data[freq_data['Phi'] == phi]
            if phi_data.empty:
                print(f"No data for Phi={phi}° at Frequency={freq} GHz in {file_path}")
                continue

            # Sort by Theta and convert to radians
            phi_data = phi_data.sort_values(by='Theta')
            theta_deg = phi_data['Theta'].values % 360
            theta_rad = np.deg2rad(theta_deg)

            # For each sublist in REQUIRED_GAIN_GROUPS, create a separate figure
            # but do not put "Group X" in the figure name/title.
            # Instead, we can name them based on the first column or simply number them.
            for sublist_index, gain_group in enumerate(REQUIRED_GAIN_GROUPS, start=1):
                # Find which columns from this sublist exist
                available_columns = [c for c in gain_group if c in phi_data.columns]
                if not available_columns:
                    continue

                plt.figure(figsize=(8, 6))
                ax = plt.subplot(111, polar=True)

                # Plot each column
                for col in available_columns:
                    ax.plot(theta_rad, phi_data[col].values, label=col)

                # Customize the plot
                ax.set_theta_zero_location('N')  # 0° at the top
                ax.set_theta_direction(-1)       # degrees increase clockwise
                ax.grid(True)
                ax.set_title(f'Freq: {freq} GHz, Phi: {phi}°', va='bottom')
                ax.set_rlim(global_r_min, global_r_max)
                ax.set_rlabel_position(225)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                plt.tight_layout()

                # Construct figure name (no "Group #")
                # Use first available column as a suffix, or just use sublist_index in parentheses if you prefer.
                suffix_col = sanitize_filename(available_columns[0])
                safe_freq = sanitize_filename(str(freq))
                safe_phi = sanitize_filename(str(phi))

                figure_filename = f"Polar_Freq_{safe_freq}GHz_Phi_{safe_phi}deg_{suffix_col}.png"
                output_path = os.path.join(output_dir, figure_filename)

                try:
                    plt.savefig(output_path, dpi=300)
                    print(f"Saved plot to {output_path}")
                except Exception as e:
                    print(f"Failed to save plot. CSV={file_path}, freq={freq}, phi={phi}: {e}")
                finally:
                    plt.close()

def generate_polar_plots(directory='.', frequency_list=None, phi_list=None):
    """
    Scan the given directory for valid CSV files and generate polar plots
    according to the REQUIRED_GAIN_GROUPS. This function can be called
    from within other scripts without relying on command-line arguments.
    """
    if frequency_list is None:
        frequency_list = FREQUENCY_LIST
    if phi_list is None:
        phi_list = PHI_LIST

    base_directory = os.path.abspath(directory)
    print(f"Scanning directory: {base_directory}")
    print(f"Frequencies to process: {frequency_list}")
    print(f"Phi angles to process: {phi_list}")

    # Find all valid CSV files
    csv_files = find_csv_files(base_directory)
    if not csv_files:
        print("No valid CSV files found.")
        return

    print(f"Found {len(csv_files)} valid CSV file(s). Processing...")

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        process_csv(csv_file, frequency_list, phi_list)

    print("Processing completed.")

# --------------------- Entry Point ---------------------
def main():
    parser = argparse.ArgumentParser(
        description='Process CSV files and generate polar plots for radiation patterns.'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Path to the directory to scan. Defaults to current directory.'
    )
    parser.add_argument(
        '--frequencies',
        nargs='+',
        type=float,
        default=FREQUENCY_LIST,
        help=f'List of frequencies to process. Default: {FREQUENCY_LIST}'
    )
    parser.add_argument(
        '--phis',
        nargs='+',
        type=float,
        default=PHI_LIST,
        help=f'List of Phi angles to process. Default: {PHI_LIST}'
    )
    args = parser.parse_args()

    generate_polar_plots(
        directory=args.directory,
        frequency_list=args.frequencies,
        phi_list=args.phis
    )

if __name__ == "__main__":
    main()