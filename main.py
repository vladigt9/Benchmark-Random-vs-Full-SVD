import pandas as pd
import numpy as np
import random
from genuineVsRand.data import createHiggsData, createHconsData
from genuineVsRand.tests import time_test
from svd_solver.test import solvers_timing
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re


def merge_txt_files(folder_path, output_file):
    # Get all txt files in the folder
    all_txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    txt_files = [f for f in all_txt_files if 'intelex' not in os.path.basename(f).lower()]
    
    with open(output_file, 'w') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r') as infile:
                outfile.write(infile.read() + '\n')

# Specify the folder containing the txt files and the output file
folder_path = 'prem_data'
output_file = 'merged_output.txt'

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

def sim_rand_svf():
    


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
        
    # Remove any unused subplots
    # for j in range(i + 1, num_rows * num_cols):
    #     fig.delaxes(axes[j])

    # plt.tight_layout()
    # plt.show()
    # Adjust layout and show plot
    # plt.savefig('results.png', dpi=300)
#
# d3_plot()



# for k in range(1):
#     random.seed(1)
#     np.random.seed(1)
#     X = np.array(np.random.rand(100,10))
    
#     # Scale half of the singluar values of the original matrix to create a new one
#     U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
#     n = (([2]*(int(len(Sigma)/2)))+([1]*(int(len(Sigma)/2))))
#     X2 = U@(np.diag(Sigma)*n)@Vt
    
#     C = np.cov(X, rowvar=False)
#     C2 = np.cov(X2, rowvar=False)

#     vmin = -0.04
#     vmax = 0.2
    
#     plt.figure(figsize=(8, 6))
#     plt.imshow(C, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
#     plt.colorbar(label='Covariance')
#     plt.title('Covariance Matrix Heatmap')
#     plt.xlabel('Variables')
#     plt.ylabel('Variables')
#     plt.xticks(np.arange(len(C)), np.arange(1, len(C)+1))
#     plt.yticks(np.arange(len(C)), np.arange(1, len(C)+1))
#     plt.savefig('0.png')
#     plt.close()

#     plt.figure(figsize=(8, 6))
#     plt.imshow(C2, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
#     plt.colorbar(label='Covariance')
#     plt.title('Covariance Matrix Heatmap')
#     plt.xlabel('Variables')
#     plt.ylabel('Variables')
#     plt.xticks(np.arange(len(C2)), np.arange(1, len(C2)+1))
#     plt.yticks(np.arange(len(C2)), np.arange(1, len(C2)+1))
#     plt.savefig('1.png')
#     plt.close()

def quick_test():
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

    df.columns = column_names
    
    
    df['sing_values'] = df['sing_values'].apply(convert_to_list)

    df = df[(df['multiplier'] == '1')]
    df = df[['r', 'c', 'sing_values', 'solver']]
    df = df[(df['solver'] == 'full') | (df['solver'] == 'randomized')]
    df['r'] = df['r'].astype(int)
    df['c'] = df['c'].astype(int)

    result = []
    for (c, r), group in df.groupby(['c', 'r']):
        if len(group) == 2:
            # Extract time lists for both rows
            row_a = group[group['solver'] == 'full'].iloc[0]
            row_b = group[group['solver'] == 'randomized'].iloc[0]
        
            # Extract time lists
            time_a = row_a['sing_values']
            time_b = row_b['sing_values']
        
            subtraction = [(np.linalg.norm(a - b))/np.linalg.norm(a) for a, b in zip(time_a, time_b)]
            # Append the result
            result.append({'r': r, 'c': c, 'subtraction': subtraction})

    # Convert the result into a DataFrame
    result_df = pd.DataFrame(result)

    print(result_df)
    
    
    
def convert_to_list(value):
    return [float(num) if '.' in num else int(num) for num in value.split('; ')]

    
# quick_test()

#100 000
