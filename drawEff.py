import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
import numpy as np

# Configure matplotlib styles up-front
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.prop_cycle'] = cycler(
    'color', ['#FF0000', '#FFAA00', '#58A500', '#00BFE9', '#2000AA', '#960096', '#808080']
) + cycler('linestyle', ['-', '-', '-', '-', '-', '-', (0, (1, 1))]) + \
    cycler('linewidth', [2]*7) + cycler('alpha', [1]*7)

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

def process_csv(file_path: str) -> None:
    """
    Process a single CSV file:
    - Skip if 'Phi' or 'Theta' columns exist.
    - If 'Efficiency . dB' does not exist but 'Efficiency' does exist, compute it.
    - Plot and save the figure in a subdirectory named after the CSV (without extension).
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    columns = data.columns.tolist()

    # Skip if 'Phi' or 'Theta' columns exist
    if 'Phi' in columns or 'Theta' in columns:
        return

    # We must have at least 'Frequency' and some form of 'Efficiency' to plot
    if 'Frequency' not in columns:
        return

    # If "Efficiency . dB" isn't present, but "Efficiency" is, compute dB values
    if 'Efficiency . dB' not in columns:
        if 'Efficiency' in columns:
            # Convert from linear to dB
            data['Efficiency . dB'] = 10.0 * np.log10(data['Efficiency'])
        else:
            # Neither Efficiency . dB nor Efficiency are present, skip
            return

    # Prepare data for plotting
    frequency = data['Frequency']
    efficiency_db = data['Efficiency . dB']

    # Create output directory (same location as original CSV, named after CSV)
    csv_filename = os.path.basename(file_path)
    csv_name, _ = os.path.splitext(csv_filename)
    output_dir = os.path.join(os.path.dirname(file_path), csv_name)
    os.makedirs(output_dir, exist_ok=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(frequency, efficiency_db, label='Efficiency . dB')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Efficiency (dB)')
    ax.set_title('Efficiency vs. Frequency')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, 'Efficiency . dB.png')
    try:
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")
    except Exception as e:
        print(f"Failed to save plot for {file_path}: {e}")
    finally:
        plt.close()

def generate_efficiency_plots(directory: str) -> None:
    """
    Scans the specified directory (and subdirectories) for CSV files
    that meet the criteria and generates efficiency plots.
    """
    directory = os.path.abspath(directory)
    csv_files = find_csv_files(directory)

    if not csv_files:
        print("No valid CSV files found.")
        return

    for csv_file in csv_files:
        process_csv(csv_file)

def main():
    """
    Entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Generate antenna efficiency plots from CSV files.')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Path to the directory to scan. Defaults to current directory.'
    )
    args = parser.parse_args()

    # Call the single function that does all the work
    generate_efficiency_plots(args.directory)

if __name__ == "__main__":
    main()