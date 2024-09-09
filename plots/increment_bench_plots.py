from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd
import ast
import matplotlib.pyplot as plt
from other.merge_txt import calculate_norm1, calculate_norm2
import numpy as np
import seaborn as sns

def increment_3d_timing_plot(filename):

    df = pd.read_csv(filename)
    df = df[['solver', 'r', 'c', 'time', 'comp']]
    
    df['time'] = np.log(df['time'])
    df = df[(df['comp'] == 2) | (df['comp'] == 4) | (df['comp'] == 6) | (df['comp'] == 8) |
            (df['comp'] == 10) | (df['comp'] == 12) | (df['comp'] == 14) | (df['comp'] == 16) |
            (df['comp'] == 18) | (df['comp'] == 20)]

    label = df['solver'].unique()
    
    unique_groups = df['c'].unique()
    unique_groups.sort()
    
    solver_colors = {
        label[0]: 'blue',
        label[1]: 'red',
        # label[2]: 'green',
    }
    
    for i, group in enumerate(unique_groups):
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        for lbl in label:
            solver_df = group_df[group_df['solver'] == lbl]
            sorted_solver_df = solver_df.sort_values(['r', 'comp'], ascending=[True, True])
            
            previous_r = None
            segment_x = []
            segment_y = []
            segment_z = []
            
            for _, row in sorted_solver_df.iterrows():
                current_r = row['r']
                if previous_r is not None and current_r != previous_r:
                    ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                    segment_x = []
                    segment_y = []
                    segment_z = []
                
                segment_x.append(row['r'])
                segment_y.append(row['comp'])
                segment_z.append(row['time'])
                previous_r = current_r
            
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl,color=solver_colors[lbl])
        
        ax.set_title(f'PCA Timing ({int(group)} Columns) (8 Threads)')
        ax.set_xlim(0, 110000)
        plt.yticks(np.arange(2, 21, 4))
        ax.set_xlabel('Number of Rows')
        ax.set_ylabel('Number of Components')
        ax.set_zlabel('Log(Fitting Time) (log(seconds))')
        ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        ax.view_init(elev=14, azim=235)
        plt.savefig(f'images/incr_{group}_8.png', dpi=300, bbox_inches=None)
        # plt.show()

def increment_3d_accuracy_plot(filename: str, columns: int, method: int):
    df = pd.read_csv(filename)
    
    df['sing_values'] = df['sing_values'].apply(ast.literal_eval)
    df['sing_values'] = df['sing_values'].apply(lambda x: [float(i) for i in x])
    
    # df = df[(df['comp'] == 2) | (df['comp'] == 4) | (df['comp'] == 6) | (df['comp'] == 8) |
    #         (df['comp'] == 10) | (df['comp'] == 12) | (df['comp'] == 14) | (df['comp'] == 16) |
    #         (df['comp'] == 18) | (df['comp'] == 20)]

    df = df[df['solver'] != 'intelex full']
    
    df1 = df[df['solver'] == 'full'].set_index(['c', 'r', 'comp'])
    df2 = df[df['solver'] == 'randomized'].set_index(['c', 'r', 'comp'])

    df = df1[['sing_values']].merge(df2[['sing_values']], left_index=True, right_index=True, suffixes=('_df1','_df2'))
    
    df = df.reset_index()
    
    df['acc1'] = df.apply(calculate_norm1, axis=1)
    df['acc2'] = df.apply(calculate_norm2, axis=1)
    
    # df['acc2'] = np.log(df['acc2'])
    
    df = df[['r', 'c', 'comp', 'acc1', 'acc2']]
    df = df[df['c'] == columns]
    
    df = df.sort_values(by=['c', 'r', 'comp'])
    
    fig, ax = plt.subplots()

    # Plot each group
    for name, group in df.groupby('comp'):
        ax.plot(group['r'], group[f'acc{method}'], marker='', label=f'{name}')

    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(f'Standard vs Randomized PCA Accuracy ({columns} columns)')
    ax.legend(title='Components')
    plt.savefig(f'images/acc_incr_{columns}.png', dpi=300)
    # plt.show()

def heatmap(filename, columns):
    df = pd.read_csv(filename)
    df = df[['solver', 'r', 'c', 'time', 'comp']]
    df = df[df['c'] == columns]

    label = df['solver'].unique()
    unique_r = sorted(df['r'].unique())
    unique_comp = sorted(df['comp'].unique())

    # Initialize the matrix for the heatmap
    heatmap_matrix = np.zeros((len(unique_r), len(unique_comp)))

    # Populate the matrix
    for i, r_value in enumerate(unique_r):
        for j, comp_value in enumerate(unique_comp):
            group_df = df[(df['r'] == r_value) & (df['comp'] == comp_value)]
            
            if len(group_df) == 2:
                solver_0_time = group_df[group_df['solver'] == label[0]]['time'].values[0]
                solver_1_time = group_df[group_df['solver'] == label[1]]['time'].values[0]
                
                heatmap_matrix[i, j] = ((solver_0_time - solver_1_time)/solver_0_time)*100
                
    colors = ['blue', 'orange']
    cmap = ListedColormap(colors)

    bounds = [-0.000000001, 0.000000001]
    norm = BoundaryNorm(boundaries=bounds, ncolors=len(colors))

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_matrix, cmap=cmap, norm=norm, cbar=True, 
                xticklabels=unique_comp, yticklabels=unique_r, annot=True, fmt=".2f")

    plt.title(f'Comparison Heatmap between {label[0]} and {label[1]} ({columns})')
    plt.xlabel('Components/Columns')
    plt.ylabel('Number of Rows')
    plt.zlabel('Number of Rows')

    plt.show()
