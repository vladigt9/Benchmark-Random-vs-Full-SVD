import pandas as pd
import numpy as np
import random
from genuineVsRand.data import createHiggsData, createHconsData
from genuineVsRand.tests import time_test
from rand_svf.test import test_svd
from svd_solver.test import solvers_timing
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re


def merge_txt_files(folder_path, output_file):
    # Get all txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    # txt_files = [f for f in all_txt_files if 'row' not in os.path.basename(f).lower()]
    
    with open(output_file, 'w') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r') as infile:
                outfile.write(infile.read() + '\n')

# Specify the folder containing the txt files and the output file
folder_path = 'prem_data_rand'
output_file = 'merged_output_rand22.txt'

# Merge the txt files
# merge_txt_files(folder_path, output_file)


random.seed(1)
np.random.seed(1)


def completeTests():
    # df_hi = createHiggsData()
    df_hc = createHconsData()
    df_hc = df_hc.iloc[:1000000, :]
    
    i = 100000
    while i <= 500000:

        for k in range(50,51, 10):
            time_test(df_hc, i, k, 100, 5, 'covariance_eigh')
        
        i+=100000

# completeTests()

def test_solvers() -> None:
    solvers_timing()
    return
    
# test_solvers()

def sim_rand_svd():
    test_svd()

# sim_rand_svd()

def t():
    
    merged_lines = []
    with open('merged_output.txt', 'r') as file:
        lines = file.readlines()
        
        # Merge every two lines into one
        for i in range(0, len(lines), 2):
            merged_line = lines[i].strip() + lines[i+1].strip()
            merged_lines.append(merged_line)
            
    df = pd.DataFrame(merged_lines)

    # Split the merged lines by comma and expand into separate columns
    df = df[0].str.split(',', expand=True)

    # Ensure all columns are treated as strings
    df = df.astype(str)

    column_names = ['solver', 'multiplier', 'r', 'c', 'time',
                                                 'expl_var', 'sing_values']  # Replace with your actual column names

    # Assign column names to the DataFrame
    df.columns = column_names
    
    # Display the DataFrame
    df = df[df['multiplier'] == '5']
    df = df[['solver', 'r', 'c', 'time']]
    # print(df)
    df['r'] = df['r'].astype(int)
    df['c'] = df['c'].astype(float)
    df['time'] = df['time'].astype(float)
    
    # df = pd.read_csv('merged_output.csv', names=['solver','multiplier', 'rows', 'columns', 'time',
    #                                              'expl_var', 'cl', 'sing_values'])
    # print(df['solver'])
    label = df['solver'].unique()
    
    unique_groups = df['c'].unique()
    unique_groups.sort()

    # Calculate the number of rows and columns for the subplots grid
    num_rows = 3
    num_cols = 3

    # Create subplots with a smaller figure size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each group in its own subplot
    for i, group in enumerate(unique_groups):
        group_df = df[df['c'] == group]
        ax = axes[i]
        ax.plot(group_df[group_df['solver']==label[0]]['r'].sort_values(ascending=True), group_df[group_df['solver']==label[0]]['time'], label=label[0])
        ax.plot(group_df[group_df['solver']==label[1]]['r'].sort_values(ascending=True), group_df[group_df['solver']==label[1]]['time'], label=label[1])
        ax.plot(group_df[group_df['solver']==label[2]]['r'].sort_values(ascending=True), group_df[group_df['solver']==label[2]]['time'], label=label[2])
        ax.set_title(f'Time needed for PCA for {int(group)} columns')
        ax.set_xlabel('Number of Rows')
        ax.set_ylabel('Time needed for fitting (seconds)')
        ax.legend()

    # Remove any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    # plt.savefig('results.png', dpi=300)

def check_accuracy():
    
    df_rand = pd.read_csv('merged_data_rand.csv')
    df_norm = pd.read_csv('merged_data.csv')
    
    df_rand['over'] = df_rand['over'].astype(int)
    df_rand['iter'] = df_rand['iter'].astype(str)
    df_rand = df_rand[df_rand['normalizer'] == 'QR']
    # df_rand = df_rand[df_rand['driver'] == 'gesdd']
    df_rand['over'] = df_rand['over']/df_rand['c']
    df_rand = df_rand[df_rand['over'] == 0.15]
    df_rand = df_rand[df_rand['iter'] == '7']
    df_rand['solver'] = ['rand2']*len(df_rand)
    
    df_norm = df_norm[df_norm['solver'] == 'full']
    
    df_rand = df_rand[['r', 'c', 'comp', 'sing_values', 'solver', 'time']]
    df_norm = df_norm[['r', 'c', 'comp', 'sing_values', 'solver', 'time']]
    
    df = pd.concat([df_norm, df_rand], axis=0, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    df['sing_values'] = df['sing_values'].str.split('; ')
    
    df = df[df['c'] == 300]
    df = df[df['r'] == 10000]
    df = df[df['comp'] == 0.1]
    
    
    df['sing_values'] = df['sing_values'].apply(lambda x: [float(i) for i in x])
    
    sing_values1 = np.array(df[df['solver'] == 'full']['sing_values'].iloc[0])
    sing_values2 = np.array(df[df['solver'] == 'rand2']['sing_values'].iloc[0])
    
    print(f"rand - full: { df[df['solver'] == 'rand2']['time'].iloc[0]-df[df['solver'] == 'full']['time'].iloc[0] }")
    print(np.linalg.norm((sing_values1 - sing_values2)/sing_values1))
    print(np.linalg.norm(sing_values1 - sing_values2)/np.linalg.norm(sing_values1))
    
# check_accuracy()

def d3_plot():
    
    df = pd.read_csv('merged_data_rand.csv')
    
    df = df.astype(str)
    df['comp'] = df['comp'].astype(float)

    # Prepare DataFrame for plotting
    # df = df[['r', 'c', 'time', 'comp', 'over', 'iter', 'normalizer', 'driver']]
    df['iter'] = df['iter'].astype(int)
    df['r'] = df['r'].astype(int)
    df['c'] = df['c'].astype(float)
    df['time'] = df['time'].astype(float)
    df['over'] = df['over'].astype(float)
    df = df[df['c'] < 400]
    df = df[df['iter'] == 7]
    # df = df[df['over'] == 'LU']
    df = df[df['driver'] == 'gesdd']
    df['sss'] = df['over']/df['c']
    # df = df[df['sss'] == 0.1]
    
    labels = df['normalizer'].unique()
    unique_groups = df['c'].unique()
    unique_groups.sort()

    solver_colors = {lbl: color for lbl, color in zip(labels, ['blue', 'red', 'green'])}
    # solver_colors = {lbl: color for lbl, color in zip(labels, ['blue', 'red'])}

    for i, group in enumerate(unique_groups):
        if i != 2:
            continue

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        for lbl in labels:
            solver_df = group_df[group_df['normalizer'] == lbl]
            sorted_solver_df = solver_df.sort_values(['sss', 'comp'], ascending=[True, True])
            
            previous_r = None
            segment_x = []
            segment_y = []
            segment_z = []
            
            for _, row in sorted_solver_df.iterrows():
                current_r = row['sss']
                if previous_r is not None and current_r != previous_r:
                    # Plot the segment
                    ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                    segment_x = []
                    segment_y = []
                    segment_z = []
                segment_x.append(row['sss'])
                segment_y.append(row['comp'])
                segment_z.append(row['time'])
                previous_r = current_r
            
            # Plot the last segment
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
        
        ax.set_title(f'Time needed for PCA for {int(group)} columns')
        ax.set_xlabel('Iter')
        ax.set_ylabel('Components')
        ax.set_zlabel('Time needed for fitting (seconds)')
        ax.legend()

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        plt.show()

d3_plot()

def calculate_norm1(row):
    sing_values = np.array(row['sing_values'])
    sing_values_df2 = np.array(row['sing_values_df2'])
    if np.linalg.norm(sing_values_df2) == 0:  # Avoid division by zero
        return np.nan
    return np.linalg.norm(sing_values_df2 - sing_values) / np.linalg.norm(sing_values_df2)
    
def calculate_norm2(row):
    sing_values = np.array(row['sing_values'])
    sing_values_df2 = np.array(row['sing_values_df2'])
    if np.linalg.norm(sing_values_df2) == 0:  # Avoid division by zero
        return np.nan
        return np.linalg.norm(sing_values_df2 - sing_values) / np.linalg.norm(sing_values_df2)
    return np.linalg.norm((sing_values_df2 - sing_values)/sing_values_df2)

def plot_accuracy():
    df_rand = pd.read_csv('merged_data_rand.csv')
    df_norm = pd.read_csv('merged_data.csv')
    
    df_rand['over'] = df_rand['over']/df_rand['c']
    df_rand['solver'] = ['rand2']*len(df_rand)
    
    df_norm = df_norm[df_norm['solver'] == 'full']
    
    df_rand = df_rand[['r', 'c', 'comp', 'sing_values', 'solver', 'normalizer', 'iter', 'over', 'time', 'driver']]
    df_norm = df_norm[['r', 'c', 'comp', 'sing_values', 'solver',]]
    
    df = df_rand.merge(df_norm[['r', 'c', 'comp', 'sing_values', 'solver']],
                       on=['r', 'c', 'comp'],
                       suffixes=('', '_df2'))
    
    df['sing_values'] = df['sing_values'].str.split('; ')
    df['sing_values_df2'] = df['sing_values_df2'].str.split('; ')
    df['sing_values'] = df['sing_values'].apply(lambda x: [float(i) for i in x])
    df['sing_values_df2'] = df['sing_values_df2'].apply(lambda x: [float(i) for i in x])

    df['acc_1'] = df.apply(calculate_norm1, axis=1)
    df['acc_2'] = df.apply(calculate_norm2, axis=1)
    
    
    df = df[['r', 'c', 'comp', 'acc_1', 'acc_2', 'iter', 'over', 'normalizer', 'time', 'driver']]
    df = df[df['c'] < 400]
    df = df[df['iter'] == 7]
    df = df[df['comp'] == 0.2]
    df = df[df['over'] == 0.15]
    df = df[df['c'] == 300]
    df = df[df['normalizer'] != 'none']
    df['acc_1'] = df['acc_1']
    print(df.head(4))
    
    labels = df['normalizer'].unique()
    unique_groups = df['c'].unique()
    unique_groups.sort()

    solver_colors = {lbl: color for lbl, color in zip(labels, ['blue', 'red'])}

    for i, group in enumerate(unique_groups):
        if i != 1:
            continue

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        for lbl in labels:
            solver_df = group_df[group_df['normalizer'] == lbl]
            sorted_solver_df = solver_df.sort_values(['over', 'comp'], ascending=[True, True])
            
            previous_r = None
            segment_x = []
            segment_y = []
            segment_z = []
            
            for _, row in sorted_solver_df.iterrows():
                current_r = row['over']
                if previous_r is not None and current_r != previous_r:
                    # Plot the segment
                    ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                    segment_x = []
                    segment_y = []
                    segment_z = []
                segment_x.append(row['over'])
                segment_y.append(row['comp'])
                segment_z.append(row['acc_1'])
                previous_r = current_r
            
            # Plot the last segment
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
        
        ax.set_title(f'Time needed for PCA for {int(group)} columns')
        ax.set_xlabel('Iter')
        ax.set_ylabel('Components')
        ax.set_zlabel('Time needed for fitting (seconds)')
        ax.legend()

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        plt.show()

# plot_accuracy()

def d3_plot():
    
    with open('merged_output.txt', 'r') as file:
        lines = file.readlines()

    # Concatenate lines that belong to the same row
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
    
    # Create DataFrame
    df = pd.DataFrame([line.split(',') for line in combined_lines])
    
    df_intel = pd.read_csv('merged_data_intel.csv')
    
    column_names = ['solver', 'r', 'c', 'comp', 'time',
                                                 'expl_var', 'sing_values']  # Replace with your actual column names
    
    df.columns = column_names
    
    df_intel.columns = column_names
    df = df.astype(str)
    df_intel = df_intel.astype(str)
    df['comp'] = df['comp'].astype(float)
    df = df[df['comp'] < 0.65]
    
    df = pd.concat([df, df_intel], axis=0, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    # print(df)
    # Ensure all columns are treated as strings


    # Assign column names to the DataFrame
    
    # Display the DataFrame
    df = df[['solver', 'r', 'c', 'time', 'comp']]
    # print(df)
    df['r'] = df['r'].astype(int)
    df['c'] = df['c'].astype(float)
    df['comp'] = df['comp'].astype(float)
    df['time'] = df['time'].astype(float)
    df = df[df['r'] < 100000]
    df = df[df['comp'] < 0.65]
    # df = df[df['solver'] != 'randomized']
    # print(df)
    # df = pd.read_csv('merged_output.csv', names=['solver','multiplier', 'rows', 'columns', 'time',
    #                                              'expl_var', 'cl', 'sing_values'])
    # print(df['solver'])
    label = df['solver'].unique()
    
    unique_groups = df['c'].unique()
    unique_groups.sort()

    # Calculate the number of rows and columns for the subplots grid
    num_rows = 4
    num_cols = 3
    
    solver_colors = {
        label[0]: 'blue',
        label[1]: 'red',
        label[2]: 'green',
    }
    
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10), subplot_kw={'projection': '3d'})
    # axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each group in its own subplot
    for i, group in enumerate(unique_groups):
        if i != 5:  
            continue
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        # ax = axes[i]
        for lbl in label:
            solver_df = group_df[group_df['solver'] == lbl]
            sorted_solver_df = solver_df.sort_values(['r', 'comp'], ascending=[True, True])
            
            # ax.scatter(sorted_solver_df['r'], sorted_solver_df['comp'], sorted_solver_df['time'], label=lbl)
        
            previous_r = None
            segment_x = []
            segment_y = []
            segment_z = []
            
            for _, row in sorted_solver_df.iterrows():
                current_r = row['r']
                if previous_r is not None and current_r != previous_r:
                    # Plot the segment
                    ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                    segment_x = []
                    segment_y = []
                    segment_z = []
                segment_x.append(row['r'])
                segment_y.append(row['comp'])
                segment_z.append(row['time'])
                previous_r = current_r
            
            # Plot the last segment
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl)
        
        ax.set_title(f'Time needed for PCA for {int(group)} columns')
        ax.set_xlabel('Number of Rows')
        ax.set_ylabel('Components/Columns (%)')
        ax.set_zlabel('Time needed for fitting (seconds)')
        ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        plt.show()

# d3_plot()
