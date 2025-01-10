import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import argparse

# Define relevant columns to process
RELEVANT_COLUMNS = ["Gain . dB", "Polar LC . Amp dB", "Polar RC . Amp dB"]

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Process CSV files and plot peak values.')
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

def process_csv(file_path):
    """
    Process a single CSV file:
    - Check for 'Frequency', 'Phi', 'Theta' columns.
    - Check for relevant columns: "Gain . dB", "Polar LC . Amp dB", "Polar RC . Amp dB".
    - For each relevant column, find the peak value across Phi and Theta for each Frequency.
    - Plot the peak values vs Frequency and save the plots.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    columns = data.columns.tolist()

    # Check for required columns
    required_main_columns = {'Frequency', 'Phi', 'Theta'}
    if not required_main_columns.issubset(columns):
        return  # Skip this file if main columns are missing

    # Identify which relevant columns are present in the CSV
    present_relevant_columns = [col for col in RELEVANT_COLUMNS if col in columns]
    if not present_relevant_columns:
        return  # No relevant columns to process

    # Prepare data by grouping by 'Frequency' and finding the max for each relevant column
    try:
        grouped = data.groupby('Frequency')[present_relevant_columns].max().reset_index()
    except Exception as e:
        print(f"Failed to group data in {file_path}: {e}")
        return

    # Create output directory
    csv_filename = os.path.basename(file_path)
    csv_name, _ = os.path.splitext(csv_filename)
    output_dir = os.path.join(os.path.dirname(file_path), csv_name)
    os.makedirs(output_dir, exist_ok=True)

    # Plot each relevant column
    for col in present_relevant_columns:
        plt.figure(figsize=(6, 4)) 

        plt.plot(grouped['Frequency'], grouped[col], label=f'Peak {col}')

        plt.xlabel('Frequency')
        plt.ylabel(col)
        plt.title(f'Peak {col} vs Frequency')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # Save the figure
        safe_col_name = col.replace('/', '_').replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        output_path = os.path.join(output_dir, f'peak_{safe_col_name}.png')
        try:
            plt.savefig(output_path, dpi=300)
            print(f"Saved plot to {output_path}")
        except Exception as e:
            print(f"Failed to save plot for {file_path}, column {col}: {e}")
        finally:
            plt.close()

def main():
    csv_files = find_csv_files(base_directory)
    if not csv_files:
        print("No valid CSV files found.")
        return

    for csv_file in csv_files:
        process_csv(csv_file)

if __name__ == "__main__":
    main()