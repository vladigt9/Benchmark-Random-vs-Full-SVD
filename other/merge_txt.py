import glob
import os
import re
import pandas as pd
import numpy as np

def merge_txt_files(folder_path, output_file):
    # Get all txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    # txt_files = [f for f in all_txt_files if 'row' not in os.path.basename(f).lower()]
    
    with open(output_file, 'w') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r') as infile:
                outfile.write(infile.read() + '\n')

def read_csv():
    df = pd.read_csv('merged_output_ran_8.csv', sep=',')
    
    columns = ['row','r','c','comp','time','n_over','n_iter','norm','driver','sing_values']
    
    df.columns = columns
    
    df['sing_values'] = df['sing_values'].str.split('; ')
    df['sing_values'] = df['sing_values'].apply(lambda x: [float(i) for i in x])
    
    
    df.to_csv('merged_data_8_rand.csv', index=False)

def calculate_norm1(row):
    sing_values = np.array(row['sing_values'])
    sing_values_df2 = np.array(row['sing_values_df2'])
    if np.linalg.norm(sing_values_df2) == 0:  # Avoid division by zero
        return np.nan
    return np.linalg.norm(sing_values_df2 - sing_values) / np.linalg.norm(sing_values_df2)
    
def calculate_norm2(row):
    sing_values = np.array(row['sing_values'])
    sing_values_df2 = np.array(row['sing_values_df2'])
    if np.linalg.norm(sing_values_df2) == 0:
        return np.nan
    return np.linalg.norm((sing_values_df2 - sing_values)/sing_values_df2)

def txt_to_csv(import_file_name, final_name):
    with open(import_file_name, 'r') as file:
        lines = file.readlines()

    combined_lines = []
    current_line = ""
    for line in lines:
        line = line.strip()
        if re.match(r'^[rf]', line):
            if current_line:
                combined_lines.append(current_line)
            current_line = line
        else:
            current_line += " " + line
    if current_line:
        combined_lines.append(current_line)

    df = pd.DataFrame([line.split(',') for line in combined_lines])

    column_names = ['solver', 'r', 'c', 'comp', 'time', 'expl_var', 'sing_values']

    df.columns = column_names

    df.to_csv(final_name, index=False)
