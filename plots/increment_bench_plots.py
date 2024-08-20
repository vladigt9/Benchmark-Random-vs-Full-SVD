import pandas as pd
import matplotlib.pyplot as plt
from other.merge_txt import calculate_norm1, calculate_norm2

def increment_3d_timing_plot(file_name):

    # Assign column names to the DataFrame
    df = pd.read_csv(file_name)
    # Display the DataFrame
    df = df[['solver', 'r', 'c', 'time', 'comp']]
    # print(df)
    df['r'] = df['r'].astype(int)
    df['c'] = df['c'].astype(float)
    df['comp'] = df['comp'].astype(float)
    df['time'] = df['time'].astype(float)
    df = df[df['r'] < 105000]
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
    }
    
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10), subplot_kw={'projection': '3d'})
    # axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each group in its own subplot
    for i, group in enumerate(unique_groups):
        if i != 0:  
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

def increment_3d_accuracy_plot(filename_rand, filename_full):
    df_rand = pd.read_csv(filename_rand)
    df_norm = pd.read_csv(filename_full)
    
    df_norm['sing_values'] = df_norm['sing_values'].str.split('; ')
    df_norm['sing_values'] = df_norm['sing_values'].apply(lambda x: [float(i) for i in x])
    # df_rand['sing_values'] = df_rand['sing_values'].apply(ast.literal_eval)
    
    df_rand['over'] = df_rand['over']/df_rand['c']
    df_rand['solver'] = ['rand2']*len(df_rand)
    
    df_norm = df_norm[df_norm['c'] == 1000]
    df_norm = df_norm[df_norm['r'] == 50000]
    
    df_norm = df_norm[df_norm['solver'] == 'full']
    
    df_rand = df_rand[['r', 'c', 'comp', 'sing_values', 'solver', 'normalizer', 'iter', 'over', 'time', 'driver']]
    df_norm = df_norm[['r', 'c', 'comp', 'sing_values', 'solver',]]
    
    df = df_rand.merge(df_norm[['r', 'c', 'comp', 'sing_values', 'solver']],
                       on=['r', 'c', 'comp'],
                       suffixes=('', '_df2'))
    
    # df['sing_values'] = df['sing_values'].str.split('; ')
    # df['sing_values_df2'] = df['sing_values_df2'].str.split('; ')
    # df['sing_values'] = df['sing_values'].apply(lambda x: [float(i) for i in x])
    # df['sing_values_df2'] = df['sing_values_df2'].apply(lambda x: [float(i) for i in x])

    df['acc_1'] = df.apply(calculate_norm1, axis=1)
    df['acc_2'] = df.apply(calculate_norm2, axis=1)
    
    # df['acc_1'] = np.log(df['acc_1'])
    # df['acc_2'] = np.log(df['acc_2'])
    
    
    df = df[['r', 'c', 'comp', 'acc_1', 'acc_2', 'iter', 'over', 'normalizer', 'time', 'driver']]
    # df = df[df['c'] < 400]
    # df = df[df['iter'] == 7]
    # df = df[df['comp'] == 0.2]
    # df = df[df['over'] == 0.15]
    # df = df[df['c'] == 300]
    # print(df)
    df = df[df['normalizer'] != 'none']
    # df['acc_1'] = df['acc_1']
    
    labels = df['normalizer'].unique()
    unique_groups = df['c'].unique()
    unique_groups.sort()

    solver_colors = {lbl: color for lbl, color in zip(labels, ['blue', 'red'])}

    for i, group in enumerate(unique_groups):
        if i != 0:
            continue

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        group_df = df[df['c'] == group]
        for lbl in labels:
            solver_df = group_df[group_df['normalizer'] == lbl]
            sorted_solver_df = solver_df.sort_values(['over', 'iter'], ascending=[True, True])
            
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
                segment_y.append(row['iter'])
                segment_z.append(row['acc_2'])
                previous_r = current_r
            
            # Plot the last segment
            if segment_x:
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
        
        ax.set_title(f'SVD Accuracy on 50000x1000 matrix')
        ax.set_xlabel('oversamples')
        ax.set_ylabel('iterations')
        ax.set_zlabel('Accuracy (Lower better)')
        ax.legend()

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        plt.show()

# def check_accuracy():
    
#     df_rand = pd.read_csv('merged_data_rand.csv')
#     df_norm = pd.read_csv('merged_data.csv')
    
#     df_rand['over'] = df_rand['over'].astype(int)
#     df_rand['iter'] = df_rand['iter'].astype(str)
#     df_rand = df_rand[df_rand['normalizer'] == 'QR']
#     # df_rand = df_rand[df_rand['driver'] == 'gesdd']
#     df_rand['over'] = df_rand['over']/df_rand['c']
#     df_rand = df_rand[df_rand['over'] == 0.15]
#     df_rand = df_rand[df_rand['iter'] == '7']
#     df_rand['solver'] = ['rand2']*len(df_rand)
    
#     df_norm = df_norm[df_norm['solver'] == 'full']
    
#     df_rand = df_rand[['r', 'c', 'comp', 'sing_values', 'solver', 'time']]
#     df_norm = df_norm[['r', 'c', 'comp', 'sing_values', 'solver', 'time']]
    
#     df = pd.concat([df_norm, df_rand], axis=0, ignore_index=True)
#     df.reset_index(drop=True, inplace=True)
#     df['sing_values'] = df['sing_values'].str.split('; ')
    
#     df = df[df['c'] == 300]
#     df = df[df['r'] == 10000]
#     df = df[df['comp'] == 0.1]
    
    
#     df['sing_values'] = df['sing_values'].apply(lambda x: [float(i) for i in x])
    
#     sing_values1 = np.array(df[df['solver'] == 'full']['sing_values'].iloc[0])
#     sing_values2 = np.array(df[df['solver'] == 'rand2']['sing_values'].iloc[0])
    
#     print(f"rand - full: { df[df['solver'] == 'rand2']['time'].iloc[0]-df[df['solver'] == 'full']['time'].iloc[0] }")
#     print(np.linalg.norm((sing_values1 - sing_values2)/sing_values1))
#     print(np.linalg.norm(sing_values1 - sing_values2)/np.linalg.norm(sing_values1))
