import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

from other.merge_txt import calculate_norm1, calculate_norm2


def plot_timing_sparse(filename, rows,):
    df = pd.read_csv(filename)
    df = df[['solver', 'r', 'c', 'time', 'density', 'comp']]
    # df = df[df['density'] == density]
    df = df[df['r'] == rows]
    df['density'] = np.log(df['density']*100)
    
    df['time'] = np.log(df['time'])

    label = df['solver'].unique()
    label.sort()
    
    unique_groups = df['c'].unique()
    unique_groups.sort()
    
    solver_colors = {
        label[0]: 'blue',
        label[1]: 'red',
    }
    
    for i, group in enumerate(unique_groups):
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        for lbl in label:
            solver_df = group_df[group_df['solver'] == lbl]
            sorted_solver_df = solver_df.sort_values(['density', 'comp'], ascending=[True, True])
            
            previous_r = None
            segment_x = []
            segment_y = []
            segment_z = []
            
            for _, row in sorted_solver_df.iterrows():
                current_r = row['density']
                if previous_r is not None and current_r != previous_r:
                    ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                    segment_x = []
                    segment_y = []
                    segment_z = []
                
                segment_x.append(row['density'])
                segment_y.append(row['comp'])
                segment_z.append(row['time'])
                previous_r = current_r
            
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl,color=solver_colors[lbl])
        
        ax.set_title(f'Sparse SVD Timing ({int(rows)} x {int(group)} matrix)')
        # ax.set_xlim(70000, 130000)
        ax.set_xlabel('Log(density)')
        ax.set_ylabel('Number of Components')
        ax.set_zlabel('Log(Fitting Time) (log(seconds))')
        ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        ax.view_init(elev=14, azim=240)
        plt.savefig(f'images/8_{group}_{rows}.png', dpi=300, bbox_inches=None)
        # plt.show()

def sparse_3d_acc_plot(filename_full: str, acc_method: int, log: bool, c):
    df = pd.read_csv(filename_full)
    dff = df[df['solver'] == 'randomized']
    dfr = df[df['solver'] == 'arpack']
    
    dff['sing_values'] = dff['sing_values'].apply(ast.literal_eval)
    dff['sing_values'] = dff['sing_values'].apply(lambda x: [float(i) for i in x])
    dfr['sing_values'] = dfr['sing_values'].apply(ast.literal_eval)
    dfr['sing_values'] = dfr['sing_values'].apply(lambda x: [float(i) for i in x])
    
    dfr = dfr[['r', 'c', 'comp', 'sing_values', 'solver', 'density']]
    dff = dff[['r', 'c', 'comp', 'sing_values', 'solver', 'density']]
    
    df = dfr.merge(dff[['r', 'c', 'comp', 'sing_values', 'solver', 'density']],
                       on=['r', 'c', 'comp', 'density'],
                       suffixes=('_df1', '_df2'))
    
    df['acc1'] = df.apply(calculate_norm1, axis=1)
    df['acc2'] = df.apply(calculate_norm2, axis=1)
    
    if log:
        df['acc1'] = np.log(df['acc1'])
        df['acc2'] = np.log(df['acc2'])
    
    df = df[['r', 'c', 'comp', 'acc1', 'acc2', 'density']]
    
    df = df[df['c'] == c]
    df = df.sort_values(['density'], ascending=[True])
    
    unique_groups = df['c'].unique()
    unique_groups.sort()


    labels = df['density'].unique()
    solver_colors = {lbl: color for lbl, color in zip(labels, ['blue', 'red', 'green', 'cyan', 'yellow', 'purple'])}
    
    for i, group in enumerate(unique_groups):

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        for lbl in labels:
            solver_df = group_df[group_df['density'] == lbl]
            sorted_solver_df = solver_df.sort_values(['r', 'comp'], ascending=[True, True])
            
            previous_r = None
            segment_x = []
            segment_y = []
            segment_z = []
            
            for _, row in sorted_solver_df.iterrows():
                current_r = row['r']
                if previous_r is not None and current_r != previous_r:
                    # Plot the segment with the color corresponding to the density label
                    ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                    segment_x = []
                    segment_y = []
                    segment_z = []
                segment_x.append(row['r'])
                segment_y.append(row['comp'])
                segment_z.append(row[f'acc{acc_method}'])
                previous_r = current_r
            
            # Plot the last segment
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
        
        ax.set_title(f'Sparse Randomized SVD Accuracy ({group} columns)')
        ax.set_xlabel('Number of Rows')
        ax.set_ylabel('Number of Components')
        ax.set_zlabel('Relative Error (%)')
        plt.xticks(np.arange(40000, 61000, 10000))

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        plt.legend(unique_labels.values(), unique_labels.keys(), title='Density')

        plt.tight_layout()
        ax.view_init(elev=10, azim=220)
        plt.savefig(f'images/sparse_{group}.png', dpi=300, bbox_inches=None)
        plt.show()
