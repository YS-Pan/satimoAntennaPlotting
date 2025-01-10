import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
import numpy as np

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Process CSV files and plot Polar Amp peak-to-peak data.')
parser.add_argument(
    'directory',
    nargs='?',
    default='.',
    help='Path to the directory to scan. Defaults to current directory.'
)
args = parser.parse_args()
base_directory = os.path.abspath(args.directory)

# Configure matplotlib styles
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

plt.rcParams['axes.prop_cycle'] = cycler(
    'color', ['#FF0000', '#FFAA00', '#58A500', '#00BFE9', '#2000AA', '#960096', '#808080']
) + cycler('linestyle', ['-']*7) + cycler('linewidth', [2]*7) + cycler('alpha', [1]*7)

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

def expand_phi(data):
    """
    Expand the phi data from 0-180 to 0-360 degrees by reflecting theta.
    """
    # Create a copy where theta is negated and phi is incremented by 180
    data_neg_theta = data.copy()
    data_neg_theta['Theta'] = -data_neg_theta['Theta']
    data_neg_theta['Phi'] = data_neg_theta['Phi'] + 180
    # Ensure Phi wraps around at 360
    data_neg_theta['Phi'] = data_neg_theta['Phi'] % 360
    # Combine the original and the extended data
    combined_data = pd.concat([data, data_neg_theta], ignore_index=True)
    return combined_data

def sanitize_filename(name):
    """
    Sanitize the column name to create a filesystem-safe filename.
    """
    return "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in name).replace(' ', '_')

def process_csv(file_path):
    """
    Process a single CSV file:
    - Check for 'Frequency', 'Phi', 'Theta' columns.
    - Identify and process relevant columns: containing 'Polar LC . Amp dB' or 'Polar RC . Amp dB'.
    - Expand phi to 0-360 degrees.
    - Filter theta between 10 and 85 degrees.
    - For each relevant column, compute peak-to-peak and trimmed peak-to-peak vs Frequency.
    - Plot and save the results, including annotations for max/min values within a specific frequency range.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    # Normalize column names to handle case sensitivity
    data.columns = [col.strip() for col in data.columns]
    columns_lower = [col.lower() for col in data.columns]

    # Check for required columns (case-insensitive)
    required_main_columns = {'frequency', 'phi', 'theta'}
    available_main_columns = set(columns_lower)
    if not required_main_columns.issubset(available_main_columns):
        missing = required_main_columns - available_main_columns
        print(f"Skipping {file_path}: Missing required columns {missing}.")
        return  # Skip this file if main columns are missing

    # Create a mapping from lower case to original case
    col_mapping = {col.lower(): col for col in data.columns}

    # Identify relevant columns containing 'Polar LC . Amp dB' or 'Polar RC . Amp dB'
    relevant_keywords = ['polar lc . amp db', 'polar rc . amp db']
    relevant_columns = [col for col in data.columns if any(keyword in col.lower() for keyword in relevant_keywords)]

    if not relevant_columns:
        print(f"No relevant Polar Amp columns found in {file_path}. Skipping.")
        return  # No relevant columns to process

    # Expand phi to 0-360 degrees
    data_expanded = expand_phi(data)

    # Filter theta between 10 and 85 degrees
    data_filtered = data_expanded[(data_expanded['Theta'] >= 10) & (data_expanded['Theta'] <= 85)]

    if data_filtered.empty:
        print(f"No data points in {file_path} after filtering theta between 10 and 85 degrees.")
        return

    # Create output directory
    csv_filename = os.path.basename(file_path)
    csv_name, _ = os.path.splitext(csv_filename)
    output_dir = os.path.join(os.path.dirname(file_path), csv_name)
    os.makedirs(output_dir, exist_ok=True)

    # Define the frequency range for annotations
    freq_min_annot = 2.02
    freq_max_annot = 2.3

    # Iterate over each relevant column
    for column in relevant_columns:
        try:
            # Ensure the column is numeric
            data_filtered[column] = pd.to_numeric(data_filtered[column], errors='coerce')
            data_clean = data_filtered.dropna(subset=['Frequency', column])
        except Exception as e:
            print(f"Failed to process column {column} in {file_path}: {e}")
            continue  # Skip this column if processing fails

        if data_clean.empty:
            print(f"No valid numeric data in column {column} of {file_path}. Skipping.")
            continue

        # Group data by Frequency
        grouped = data_clean.groupby('Frequency')[column]

        # Compute peak-to-peak (max - min) per frequency
        p2p = grouped.agg(lambda x: x.max() - x.min())

        # Compute trimmed peak-to-peak (10% trimmed)
        trimmed_p2p = grouped.apply(lambda x: np.percentile(x, 95) - np.percentile(x, 5))

        # Prepare the plot
        plt.figure(figsize=(10, 7))

        plt.plot(p2p.index, p2p.values, label='Peak-to-Peak', marker='o')
        plt.plot(trimmed_p2p.index, trimmed_p2p.values, label='Trimmed Peak-to-Peak (5%)', marker='s')

        plt.xlabel('Frequency (GHz)')
        plt.ylabel(f'{column}')
        plt.title(f'Peak-to-Peak vs Frequency for {csv_name} - {column}')
        plt.grid(True)
        plt.legend()

        # Annotation within the specified frequency range
        freq_mask = (p2p.index >= freq_min_annot) & (p2p.index <= freq_max_annot)
        if freq_mask.any():
            # Max annotation
            max_p2p = p2p[freq_mask].max()
            max_freq = p2p[freq_mask].idxmax()

            plt.scatter(max_freq, max_p2p, color='red', zorder=5)
            plt.annotate(
                f'Max: {max_p2p:.2f}\nFreq: {max_freq:.2f} GHz',
                xy=(max_freq, max_p2p),
                xytext=(max_freq, max_p2p + 0.05 * max_p2p),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10,
                ha='center'
            )

            # Min annotation
            min_p2p = p2p[freq_mask].min()
            min_freq = p2p[freq_mask].idxmin()

            plt.scatter(min_freq, min_p2p, color='blue', zorder=5)
            plt.annotate(
                f'Min: {min_p2p:.2f}\nFreq: {min_freq:.2f} GHz',
                xy=(min_freq, min_p2p),
                xytext=(min_freq, min_p2p - 0.05 * abs(min_p2p)),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                fontsize=10,
                ha='center'
            )

            # Print annotated values to the console
            print(f"File: {csv_name} | Column: {column}")
            print(f"Peak-to-Peak Max within {freq_min_annot}-{freq_max_annot} GHz: {max_p2p:.2f} at {max_freq:.2f} GHz")
            print(f"Peak-to-Peak Min within {freq_min_annot}-{freq_max_annot} GHz: {min_p2p:.2f} at {min_freq:.2f} GHz\n")
        else:
            print(f"No data points in {csv_name} for column '{column}' within {freq_min_annot}-{freq_max_annot} GHz.")

        plt.tight_layout()

        # Sanitize column name for filename
        safe_column_name = sanitize_filename(column)
        output_filename = f"pk2pk_{safe_column_name}.png"
        output_path = os.path.join(output_dir, output_filename)

        try:
            plt.savefig(output_path, dpi=300)
            print(f"Saved plot to {output_path}\n")
        except Exception as e:
            print(f"Failed to save plot for column {column} in {file_path}: {e}")
        finally:
            plt.close()

def main():
    csv_files = find_csv_files(base_directory)
    if not csv_files:
        print("No valid CSV files found.")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        process_csv(csv_file)

if __name__ == "__main__":
    main()