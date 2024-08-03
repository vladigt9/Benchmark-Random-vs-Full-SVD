import pandas as pd
import numpy as np
import random
from genuineVsRand.data import createHiggsData, createHconsData
from genuineVsRand.tests import time_test
from svd_solver.test import solvers_timing
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def merge_txt_files(folder_path, output_file):
    # Get all txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
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


def t():
    
    df = pd.read_csv('merged_output.csv', names=['solver', 'r', 'c', 'time'])
    label = df['solver'].unique()
    
    unique_groups = df['c'].unique()
    unique_groups.sort()

    # Calculate the number of rows and columns for the subplots grid
    num_rows = 3
    num_cols = 3

    # Create subplots with a smaller figure size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each group in its own subplot
    for i, group in enumerate(unique_groups):
        group_df = df[df['c'] == group]
        ax = axes[i]
        ax.plot(group_df[group_df['solver']==label[0]]['r'].sort_values(ascending=True), group_df[group_df['solver']==label[0]]['time'], label=label[0])
        ax.plot(group_df[group_df['solver']==label[1]]['r'].sort_values(ascending=True), group_df[group_df['solver']==label[1]]['time'], label=label[1])
        ax.plot(group_df[group_df['solver']==label[2]]['r'].sort_values(ascending=True), group_df[group_df['solver']==label[2]]['time'], label=label[2])
        ax.set_title(f'Group {group}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.legend()

    # Remove any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# t()


for k in range(1):
    random.seed(1)
    np.random.seed(1)
    X = np.array(np.random.rand(100,10))
    
    # Scale half of the singluar values of the original matrix to create a new one
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    n = (([2]*(int(len(Sigma)/2)))+([1]*(int(len(Sigma)/2))))
    X2 = U@(np.diag(Sigma)*n)@Vt
    
    C = np.cov(X, rowvar=False)
    C2 = np.cov(X2, rowvar=False)

    vmin = -0.04
    vmax = 0.2
    
    plt.figure(figsize=(8, 6))
    plt.imshow(C, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Covariance')
    plt.title('Covariance Matrix Heatmap')
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.xticks(np.arange(len(C)), np.arange(1, len(C)+1))
    plt.yticks(np.arange(len(C)), np.arange(1, len(C)+1))
    plt.savefig('0.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(C2, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Covariance')
    plt.title('Covariance Matrix Heatmap')
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.xticks(np.arange(len(C2)), np.arange(1, len(C2)+1))
    plt.yticks(np.arange(len(C2)), np.arange(1, len(C2)+1))
    plt.savefig('1.png')
    plt.close()
