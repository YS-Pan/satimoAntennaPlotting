import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import argparse

# -------------------- Global Variables --------------------
THETA_RANGES = [80]  # Examples: [80], [80, 85], or [[10,20], 180, [40,60]]
PHI_RANGES = [[0, 360]]  # Examples: [180], or [[0,180], [180,360]]
TARGET_COLUMNS = ["Polar LC . Amp dB", "Polar RC . Amp dB","Axial Ratio (dB)"]  # e.g. ["Efficiency", "Gain Theta. dB", ...]
ANNOTATION_FREQ_RANGES = [[1.51, 1.53], [1.66, 1.68]]

def is_valid_csv(filename):
    """
    Check if the file is a .csv file and does not start with 'sliced_'.
    """
    return filename.lower().endswith('.csv') and not os.path.basename(filename).startswith('sliced_')

def find_csv_files(directory):
    """
    Traverse the directory and its subdirectories to find all valid CSV files.
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_valid_csv(file):
                csv_files.append(os.path.join(root, file))
    return csv_files

def expand_phi(data):
    """
    Expand the phi data from 0–180 to 0–360 degrees by reflecting theta.
    Assumes that for each row, if Theta is present, we mirror with Theta = -Theta and Phi+180.
    """
    data_neg_theta = data.copy()
    data_neg_theta['Theta'] = -data_neg_theta['Theta']
    data_neg_theta['Phi'] = data_neg_theta['Phi'] + 180
    # Ensure Phi wraps around at 360
    data_neg_theta['Phi'] = data_neg_theta['Phi'] % 360
    # Combine original and mirrored data
    combined_data = pd.concat([data, data_neg_theta], ignore_index=True)
    return combined_data

def sanitize_filename(name):
    """
    Sanitize the column name to create a filesystem-safe filename.
    """
    return "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in name).replace(' ', '_')

def configure_matplotlib():
    """
    Configure matplotlib styles.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.prop_cycle'] = cycler(
        'color', ['#FF0000', '#FFAA00', '#58A500', '#00BFE9', '#2000AA', '#960096', '#808080']
    ) + cycler('linestyle', ['-'] * 7) + cycler('linewidth', [2] * 7) + cycler('alpha', [1] * 7)

def find_nearest_value(available_values, target):
    """
    Find the single nearest value from 'available_values' to 'target'.
    """
    return min(available_values, key=lambda x: abs(x - target))

def filter_data_by_ranges(data, column_name, ranges_list):
    """
    Filter a DataFrame 'data' by 'column_name' using a list of ranges.
    
    Each element in 'ranges_list' can be:
      - A single numeric value (e.g. 85), for which the nearest value is taken.
      - A list [low, high], for which we filter between low <= value <= high.
      
    The final returned DataFrame is the UNION of all subsets.
    """
    if not ranges_list:
        # If no ranges are given, return empty DataFrame
        return data.iloc[0:0]
    
    result_df = pd.DataFrame(columns=data.columns)
    available_vals = data[column_name].unique()

    for item in ranges_list:
        if isinstance(item, (int, float)):
            # Single numeric => find nearest
            nearest_val = find_nearest_value(available_vals, item)
            subset = data[data[column_name] == nearest_val]
            result_df = pd.concat([result_df, subset], ignore_index=True)
        elif isinstance(item, list) and len(item) == 2:
            # Range [start, end]
            start_val, end_val = sorted(item)
            subset = data[(data[column_name] >= start_val) & (data[column_name] <= end_val)]
            result_df = pd.concat([result_df, subset], ignore_index=True)
        else:
            # Invalid format => skip
            pass
    
    result_df.drop_duplicates(inplace=True)
    return result_df

def process_csv(file_path):
    """
    Process a single CSV file:
    - Check for 'Frequency', 'Phi', 'Theta' columns.
    - Identify relevant columns from TARGET_COLUMNS.
    - Filter data by THETA_RANGES and PHI_RANGES.
    - Expand Phi to 0–360 degrees if needed.
    - Compute max, average, and min per frequency for each relevant column.
    - Plot and annotate the results within each frequency range of ANNOTATION_FREQ_RANGES.
    - Add a yellow shadow region for each frequency range.
    - Save the resulting plot to disk.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    columns = data.columns.tolist()
    required_main_columns = {'Frequency', 'Phi', 'Theta'}

    # Ensure all columns we need are present
    if not required_main_columns.issubset(set(columns)):
        print(f"Skipping '{file_path}': Missing required columns {required_main_columns}.")
        return

    relevant_columns = [col for col in TARGET_COLUMNS if col in columns]
    if not relevant_columns:
        print(f"No target columns found in '{file_path}'. Skipping.")
        return

    data_filtered_theta = filter_data_by_ranges(data, 'Theta', THETA_RANGES)
    if data_filtered_theta.empty:
        print(f"No data matching THETA_RANGES={THETA_RANGES} in '{file_path}'. Skipping.")
        return

    data_filtered_phi = filter_data_by_ranges(data_filtered_theta, 'Phi', PHI_RANGES)
    if data_filtered_phi.empty:
        print(f"No data matching PHI_RANGES={PHI_RANGES} (after Theta filtering) in '{file_path}'. Skipping.")
        return

    # Expand Phi to 0–360
    data_expanded = expand_phi(data_filtered_phi)

    # Create output directory
    csv_filename = os.path.basename(file_path)
    csv_name, _ = os.path.splitext(csv_filename)
    output_dir = os.path.join(os.path.dirname(file_path), csv_name)
    os.makedirs(output_dir, exist_ok=True)

    for column in relevant_columns:
        try:
            grouped = data_expanded.groupby('Frequency')[column].agg(['max', 'mean', 'min']).reset_index()
            grouped.rename(columns={'max': 'Max', 'mean': 'Average', 'min': 'Min'}, inplace=True)
        except Exception as e:
            print(f"Failed to group data for column '{column}' in '{file_path}': {e}")
            continue

        plt.figure(figsize=(10, 7))

        plt.plot(grouped['Frequency'], grouped['Max'], label='Max', marker='o')
        plt.plot(grouped['Frequency'], grouped['Average'], label='Average', marker='s')
        plt.plot(grouped['Frequency'], grouped['Min'], label='Min', marker='^')

        plt.xlabel('Frequency (GHz)')
        plt.ylabel(column)
        plt.title(f'{column} vs Frequency for {csv_name}')
        plt.grid(True)
        plt.legend()

        # Annotate for each frequency range in ANNOTATION_FREQ_RANGES
        for freq_range in ANNOTATION_FREQ_RANGES:
            if len(freq_range) != 2:
                continue
            freq_min, freq_max = sorted(freq_range)

            # Add yellow shadow for the freq range
            plt.axvspan(freq_min, freq_max, color='yellow', alpha=0.2)

            freq_mask = (grouped['Frequency'] >= freq_min) & (grouped['Frequency'] <= freq_max)
            grouped_filtered = grouped[freq_mask]

            if not grouped_filtered.empty:
                max_val = grouped_filtered['Max'].max()
                max_row = grouped_filtered[grouped_filtered['Max'] == max_val].iloc[0]
                max_freq = max_row['Frequency']

                plt.scatter(max_freq, max_val, color='red', zorder=5)
                # Annotation with 3 decimal places:
                plt.annotate(
                    f'Max: {max_val:.3f}\nFreq: {max_freq:.3f} GHz',
                    xy=(max_freq, max_val),
                    xytext=(max_freq, max_val + (0.05 * max_val)),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=10,
                    ha='center'
                )

                avg_val = grouped_filtered['Average'].max()
                avg_row = grouped_filtered[grouped_filtered['Average'] == avg_val].iloc[0]
                avg_freq = avg_row['Frequency']

                plt.scatter(avg_freq, avg_val, color='blue', zorder=5)
                plt.annotate(
                    f'Avg Max: {avg_val:.3f}\nFreq: {avg_freq:.3f} GHz',
                    xy=(avg_freq, avg_val),
                    xytext=(avg_freq, avg_val + (0.05 * avg_val)),
                    arrowprops=dict(facecolor='blue', shrink=0.05),
                    fontsize=10,
                    ha='center'
                )

                min_val = grouped_filtered['Min'].min()
                min_row = grouped_filtered[grouped_filtered['Min'] == min_val].iloc[0]
                min_freq = min_row['Frequency']

                plt.scatter(min_freq, min_val, color='green', zorder=5)
                plt.annotate(
                    f'Min: {min_val:.3f}\nFreq: {min_freq:.3f} GHz',
                    xy=(min_freq, min_val),
                    xytext=(min_freq, min_val - (0.05 * abs(min_val))),
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    fontsize=10,
                    ha='center'
                )

                # Keep console prints at 2 decimal places:
                print(f"File: {csv_name} | Column: {column} | Freq Range: {freq_min}-{freq_max} GHz")
                print(f"Max: {max_val:.2f} at {max_freq:.2f} GHz")
                print(f"Avg Max: {avg_val:.2f} at {avg_freq:.2f} GHz")
                print(f"Min: {min_val:.2f} at {min_freq:.2f} GHz\n")
            else:
                print(f"No data points in '{csv_name}' for '{column}' within {freq_min}-{freq_max} GHz.")

        plt.tight_layout()

        safe_column_name = sanitize_filename(column)
        output_filename = f"maxAvgMin_{safe_column_name}.png"
        output_path = os.path.join(output_dir, output_filename)

        try:
            plt.savefig(output_path, dpi=300)
            print(f"Saved plot to '{output_path}'\n")
        except Exception as e:
            print(f"Failed to save plot for column '{column}' in '{file_path}': {e}")
        finally:
            plt.close()

def run_script(directory=None):
    """
    Primary function that configures matplotlib, scans for CSV files, and processes them.
    This function can be called directly by other scripts.
    """
    configure_matplotlib()

    if directory is None:
        directory = '.'
    base_directory = os.path.abspath(directory)

    csv_files = find_csv_files(base_directory)
    if not csv_files:
        print("No valid CSV files found.")
        return

    for csv_file in csv_files:
        print(f"Processing '{csv_file}'...")
        process_csv(csv_file)

    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files with expanded theta/phi ranges.')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Path to the directory to scan. Defaults to current directory.'
    )
    args = parser.parse_args()
    run_script(args.directory)