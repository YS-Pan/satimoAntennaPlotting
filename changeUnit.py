import os
import pandas as pd
import numpy as np

def process_txt_files():
    """
    Processes all .txt files in the script's directory and subdirectories.
    For each valid table:
    - Converts Frequency from Hz to GHz (rounded to 0.0001 GHz)
    - Converts Phi and Theta from radians to degrees (rounded to 0.1 degrees)
    - Saves the transformed data as a .csv file with the same name
    """

    processed_count = 0

    # Walk through all files and directories starting from the current directory
    for root, _, files in os.walk("."):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_path = os.path.join(root, file)
                try:
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()

                    # Check if the file has at least three lines
                    if len(lines) < 3:
                        continue

                    headers = lines[1].strip().split('\t')
                    data_rows = lines[2:]

                    # Check for consistent number of columns
                    if not all(len(row.strip().split('\t')) == len(headers) for row in data_rows):
                        continue

                    # Read the data into a DataFrame
                    data = pd.read_csv(txt_path, sep='\t', skiprows=1)

                    if data.empty:
                        continue

                    # Transform units
                    if 'Frequency' in data.columns:
                        data['Frequency'] = (data['Frequency'] / 1e9).round(4)  # Hz to GHz
                    if 'Phi' in data.columns:
                        data['Phi'] = np.rad2deg(data['Phi']).round(1)  # Radians to Degrees
                    if 'Theta' in data.columns:
                        data['Theta'] = np.rad2deg(data['Theta']).round(1)  # Radians to Degrees

                    # Save to CSV
                    csv_filename = os.path.splitext(file)[0] + ".csv"
                    csv_path = os.path.join(root, csv_filename)
                    data.to_csv(csv_path, index=False)

                    processed_count += 1

                except Exception:
                    # Skip files that cause any exceptions
                    continue

    print(f"Processed {processed_count} file(s).")

if __name__ == "__main__":
    process_txt_files()