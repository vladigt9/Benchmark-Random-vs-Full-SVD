import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from other.merge_txt import calculate_norm1, calculate_norm2

def rand_svd_3d_timing_plot(filename):
    df = pd.read_csv(filename)
    
    df['sing_values'] = df['sing_values'].apply(ast.literal_eval)
    
    labels = df['norm'].unique()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    solver_colors = {
        labels[0]: 'blue',
        labels[1]: 'red',
        labels[2]: 'green'
    }

    for lbl in labels:
        solver_df = df[df['norm'] == lbl]
        sorted_solver_df = solver_df.sort_values(['n_over', 'n_iter'], ascending=[True, True])
        
        previous_r = None
        segment_x = []
        segment_y = []
        segment_z = []
        
        for _, row in sorted_solver_df.iterrows():
            current_r = row['n_over']
            if previous_r is not None and current_r != previous_r:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                segment_x = []
                segment_y = []
                segment_z = []
            
            segment_x.append(row['n_over'])
            segment_y.append(row['n_iter'])
            segment_z.append(row['time'])
            previous_r = current_r
        
        if segment_x:
            ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])

    ax.set_title('Randomized SVD Timing (50 000 x 1 000 matrix) (8 Threads)') 
    ax.set_xlabel('Number of Oversamples')
    ax.set_ylabel('Number of Iterations')
    ax.set_zlabel('Fitting Time (seconds)')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {label: handle for handle, label in zip(handles, labels)}
    ax.legend(unique_labels.values(), unique_labels.keys())

    plt.tight_layout()
    ax.view_init(elev=10, azim=250)
    plt.savefig('images/solo_timing_8.png', dpi=300, bbox_inches=None)
    # plt.show()

def rand_svd_3d_accuracy_plot(filename_full: str, filename_rand:str, normalizers: int, acc_method: int, log: bool):
    dff = pd.read_csv(filename_full)
    dfr = pd.read_csv(filename_rand)
    
    dff['sing_values'] = dff['sing_values'].apply(ast.literal_eval)
    dff['sing_values'] = dff['sing_values'].apply(lambda x: [float(i) for i in x])
    dfr['sing_values'] = dfr['sing_values'].apply(ast.literal_eval)
    dfr['sing_values'] = dfr['sing_values'].apply(lambda x: [float(i) for i in x])

    # dfr['n_over'] = dfr['n_over']/dfr['c']
    dfr['solver'] = ['randomized']*len(dfr)
    
    dff = dff[dff['c'] == 1000]
    dff = dff[dff['r'] == 50000]
    dff = dff[dff['solver'] == 'full']
    
    dfr = dfr[['r', 'c', 'comp', 'sing_values', 'solver', 'norm', 'n_iter', 'n_over', 'time', 'driver']]
    dff = dff[['r', 'c', 'comp', 'sing_values', 'solver',]]
    
    df = dfr.merge(dff[['r', 'c', 'comp', 'sing_values', 'solver']],
                       on=['r', 'c', 'comp'],
                       suffixes=('_df1', '_df2'))
    
    df['acc1'] = df.apply(calculate_norm1, axis=1)
    df['acc2'] = df.apply(calculate_norm2, axis=1)
    
    if log:
        df['acc1'] = np.log(df['acc1'])
        df['acc2'] = np.log(df['acc2'])
    
    df = df[['r', 'c', 'comp', 'acc1', 'acc2', 'n_iter', 'n_over', 'norm', 'time', 'driver']]
    
    unique_groups = df['c'].unique()
    unique_groups.sort()

    if normalizers == 1:
        df = df[df['norm'] == 'QR']
        labels = df['norm'].unique()
        solver_colors = {lbl: color for lbl, color in zip(labels, ['blue'])}
    elif normalizers == 2:
        df = df[df['norm'] != 'none']
        labels = df['norm'].unique()
        solver_colors = {lbl: color for lbl, color in zip(labels, ['blue', 'red'])}
    elif normalizers == 3:
        labels = df['norm'].unique()
        solver_colors = {lbl: color for lbl, color in zip(labels, ['blue', 'red', 'green'])}

    for i, group in enumerate(unique_groups):
        if i != 0:
            continue

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        for lbl in labels:
            solver_df = group_df[group_df['norm'] == lbl]
            sorted_solver_df = solver_df.sort_values(['n_over', 'n_iter'], ascending=[True, True])
            
            previous_r = None
            segment_x = []
            segment_y = []
            segment_z = []
            
            for _, row in sorted_solver_df.iterrows():
                current_r = row['n_over']
                if previous_r is not None and current_r != previous_r:
                    # Plot the segment
                    ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                    segment_x = []
                    segment_y = []
                    segment_z = []
                segment_x.append(row['n_over'])
                segment_y.append(row['n_iter'])
                segment_z.append(row[f'acc{acc_method}'])
                previous_r = current_r
            
            # Plot the last segment
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
        
        ax.set_title(f'Randomized SVD Accuracy (50 000 x 1 000 matrix)')
        ax.set_xlabel('Number of Oversamples')
        ax.set_ylabel('Number of Iterations')
        ax.set_zlabel('Relative Error')

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        plt.legend(unique_labels.values(), unique_labels.keys(),title='Normalizer')

        plt.tight_layout()
        ax.view_init(elev=10, azim=40)
        plt.savefig('images/solo_acc_2_2.png', dpi=300, bbox_inches=None)
        # plt.show()
