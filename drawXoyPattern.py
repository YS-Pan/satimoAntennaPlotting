import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import math

# --------------------- Configuration ---------------------

# Define a global variable with the column groups
COLUMN_GROUPS = [
    ["Gain . dB", "Gain Phi. dB", "Gain Theta. dB"],
    ["Polar LC . Amp dB", "Polar RC . Amp dB"]
]

# Define the frequency list (hardcoded at the beginning)
FREQUENCY_LIST = [1.52, 1.67]

# Define the theta list for which radiation patterns will be generated
THETA_LIST = [80]  # Add desired theta angles here

# --------------------- Matplotlib Styles ---------------------

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

plt.rcParams["axes.prop_cycle"] = (
    cycler("color", ["#FF0000", "#FFAA00", "#58A500", "#00BFE9", "#2000AA", "#960096", "#808080"])
    + cycler("linestyle", ["-"] * 7)
    + cycler("linewidth", [2] * 7)
    + cycler("alpha", [1] * 7)
)

# --------------------- Helper Functions ---------------------

def is_valid_csv(filename):
    """
    Check if the file is a .csv file and does not start with 'sliced_'.
    """
    return filename.lower().endswith(".csv") and not os.path.basename(filename).startswith("sliced_")

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
    return "".join(c if c in keep_chars else "_" for c in name)

def find_nearest_theta(desired_theta, available_thetas):
    """
    Find the nearest theta value from available_thetas to the desired_theta.
    """
    available_thetas = np.array(available_thetas)
    index = (np.abs(available_thetas - desired_theta)).argmin()
    nearest_theta = available_thetas[index]
    return nearest_theta

def compute_r_limits(all_values):
    """
    Given a list of values, compute the radial limits for the polar plots:
      - maxRange is the max value rounded up to the nearest multiple of 5
      - minRange = maxRange - 30
    If all_values is empty, return (0, 20) by default.
    """
    if not all_values:
        return (0, 20)
    max_val = max(all_values)
    max_range = int(math.ceil(max_val / 5.0) * 5)
    min_range = max_range - 30
    return (min_range, max_range)

def process_csv(file_path, frequency_list, theta_list):
    """
    Process a single CSV file:
      - Extract data for specified frequencies and each theta in theta_list.
      - If desired theta is not present, use the nearest available theta.
      - Complement phi to cover 0-360 degrees when Theta is negative (by adding 180 degrees).
      - For each sublist of COLUMN_GROUPS, create a polar plot of any columns in that sublist that exist.
      - All generated plots share the same r-limit, computed from the maximum of all plotted data in the file.
      - Save each generated figure to a subfolder named after the CSV file.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    columns = data.columns.tolist()

    # Check for required columns
    required_main_columns = {"Frequency", "Phi", "Theta"}
    if not required_main_columns.issubset(columns):
        print(f"Skipping {file_path}: Missing one of the required columns {required_main_columns}")
        return  # Skip this file if main columns are missing

    # Identify data columns (excluding 'Frequency', 'Phi', 'Theta')
    data_columns = [col for col in columns if col not in ["Frequency", "Phi", "Theta"]]

    if not data_columns:
        print(f"No data columns to plot in {file_path}")
        return

    # Filter data for the specified frequencies
    data_filtered_freq = data[data["Frequency"].isin(frequency_list)]

    if data_filtered_freq.empty:
        print(f"No data matching the specified frequencies in {file_path}")
        return

    # Get available theta values
    available_thetas = data_filtered_freq["Theta"].unique()
    available_thetas_sorted = np.sort(available_thetas)

    # Create a mapping from desired_theta to actual_theta used
    theta_mapping = {}
    for desired_theta in theta_list:
        if desired_theta in available_thetas:
            theta_mapping[desired_theta] = desired_theta
        else:
            # Find the nearest available theta
            nearest_theta = find_nearest_theta(desired_theta, available_thetas_sorted)
            theta_mapping[desired_theta] = nearest_theta
            print(f"Desired Theta={desired_theta}° not found in {file_path}. Using nearest Theta={nearest_theta}° instead.")

    # Determine global radial limits for all plots in this CSV
    # Collect all relevant data in a single list
    all_values_for_csv = []

    # We'll gather all data from the relevant frequencies, thetas, and column groups
    for desired_theta, actual_theta in theta_mapping.items():
        # Filter data for the current actual_theta
        data_theta_pos = data_filtered_freq[data_filtered_freq["Theta"] == actual_theta].copy()
        data_theta_neg = data_filtered_freq[data_filtered_freq["Theta"] == -actual_theta].copy()

        # Adjust phi for theta = -actual_theta by adding 180 degrees
        if not data_theta_neg.empty:
            data_theta_neg["Phi"] = (data_theta_neg["Phi"] + 180) % 360

        # Combine the two datasets to cover phi=0-360
        combined_data = pd.concat([data_theta_pos, data_theta_neg], ignore_index=True)
        if combined_data.empty:
            continue

        # For each frequency in the list, check each group
        for freq in frequency_list:
            data_at_freq = combined_data[combined_data["Frequency"] == freq]
            if data_at_freq.empty:
                continue

            for column_sublist in COLUMN_GROUPS:
                existing_cols = [col for col in column_sublist if col in data_at_freq.columns]
                if not existing_cols:
                    continue

                # Collect all data for these columns
                for col in existing_cols:
                    # Extend all_values_for_csv with the valid values from that column
                    all_values_for_csv.extend(data_at_freq[col].dropna().values)

    # Compute the uniform radial limits for this CSV
    min_r, max_r = compute_r_limits(all_values_for_csv)

    # Create output directory
    csv_filename = os.path.basename(file_path)
    csv_name, _ = os.path.splitext(csv_filename)
    output_dir = os.path.join(os.path.dirname(file_path), csv_name)
    os.makedirs(output_dir, exist_ok=True)

    # Now that we have uniform radial limits, generate plots
    for desired_theta, actual_theta in theta_mapping.items():
        data_theta_pos = data_filtered_freq[data_filtered_freq["Theta"] == actual_theta].copy()
        data_theta_neg = data_filtered_freq[data_filtered_freq["Theta"] == -actual_theta].copy()

        if not data_theta_pos.empty or not data_theta_neg.empty:
            # Adjust phi for negative theta
            if not data_theta_neg.empty:
                data_theta_neg["Phi"] = (data_theta_neg["Phi"] + 180) % 360

            # Merge them
            combined_data = pd.concat([data_theta_pos, data_theta_neg], ignore_index=True)

            # Sort Phi
            combined_data["Phi"] = combined_data["Phi"] % 360
            combined_data = combined_data.sort_values(by="Phi")

            if combined_data.empty:
                continue

            # Plot for each frequency
            for freq in frequency_list:
                data_at_freq = combined_data[combined_data["Frequency"] == freq]
                if data_at_freq.empty:
                    continue

                # For each group of columns, plot them together if they exist
                for column_sublist in COLUMN_GROUPS:
                    existing_cols = [col for col in column_sublist if col in data_at_freq.columns]
                    if not existing_cols:
                        continue

                    plt.figure(figsize=(8, 6))
                    ax = plt.subplot(111, polar=True)

                    phi_rad = np.deg2rad(data_at_freq["Phi"].values)

                    for col in existing_cols:
                        ax.plot(phi_rad, data_at_freq[col], label=col)

                    # Polar plot styling
                    ax.set_theta_zero_location("N")  # 0 degrees at the top
                    ax.set_theta_direction(-1)       # Degrees increase clockwise
                    ax.set_title(
                        f"{', '.join(existing_cols)} at {freq} GHz\n"
                        f"Theta={desired_theta}° (Using {actual_theta}°)",
                        va="bottom"
                    )
                    ax.grid(True)
                    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

                    # Apply the uniform radial limits
                    ax.set_rlim(min_r, max_r)

                    plt.tight_layout()

                    # Build a safe filename
                    safe_col_names = "_".join(sanitize_filename(c) for c in existing_cols)
                    figure_filename = (
                        f"XOY_{safe_col_names}_{freq}GHz_"
                        f"Theta{desired_theta}deg_(UsedTheta{actual_theta}deg).png"
                    )
                    output_path = os.path.join(output_dir, figure_filename)

                    try:
                        plt.savefig(output_path, dpi=300)
                        print(f"Saved plot to {output_path}")
                    except Exception as e:
                        print(
                            f"Failed to save plot for {file_path}, columns {existing_cols}, "
                            f"frequency {freq}, desired theta {desired_theta}: {e}"
                        )
                    finally:
                        plt.close()

def generate_plots(directory="."):
    """
    Scan the given directory for valid CSV files and process each one.
    """
    base_directory = os.path.abspath(directory)

    # Find all valid CSV files
    csv_files = find_csv_files(base_directory)
    if not csv_files:
        print("No valid CSV files found.")
        return

    print(f"Found {len(csv_files)} valid CSV file(s). Processing...")

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        process_csv(csv_file, FREQUENCY_LIST, THETA_LIST)

    print("Processing completed.")

# --------------------- Entry Point ---------------------

if __name__ == "__main__":
    # If you want to run from the command line, just call generate_plots with the desired directory
    generate_plots(".")