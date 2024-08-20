import pandas as pd
import ast
import matplotlib.pyplot as plt
from other.merge_txt import calculate_norm1, calculate_norm2

def increment_3d_timing_plot(filename):

    df = pd.read_csv(filename)
    df = df[['solver', 'r', 'c', 'time', 'comp']]

    label = df['solver'].unique()
    
    unique_groups = df['c'].unique()
    unique_groups.sort()

    # Calculate the number of rows and columns for the subplots grid
    num_rows = 4
    num_cols = 3
    
    solver_colors = {
        label[0]: 'blue',
        label[1]: 'red',
        # label[2]: 'green',
    }
    
    for i, group in enumerate(unique_groups):
        if i != 0:  
            continue
        
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
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl)
        
        ax.set_title(f'PCA timing ({int(group)} Columns)')
        ax.set_xlabel('Number of Rows')
        ax.set_ylabel('Components/Columns')
        ax.set_zlabel('Fitting time (seconds)')
        ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        plt.show()

def increment_3d_accuracy_plot(filename: str, columns: int, method: int):
    df = pd.read_csv(filename)
    
    df['sing_values'] = df['sing_values'].apply(ast.literal_eval)
    df['sing_values'] = df['sing_values'].apply(lambda x: [float(i) for i in x])

    df = df[df['solver'] != 'intelex full']
    
    df1 = df[df['solver'] == 'full'].set_index(['c', 'r', 'comp'])
    df2 = df[df['solver'] == 'randomized'].set_index(['c', 'r', 'comp'])

    df = df1[['sing_values']].merge(df2[['sing_values']], left_index=True, right_index=True, suffixes=('_df1','_df2'))
    
    df = df.reset_index()
    
    df['acc1'] = df.apply(calculate_norm1, axis=1)
    df['acc2'] = df.apply(calculate_norm2, axis=1)
    # df['comp'] = df['comp'].astype(str)
    
    df = df[['r', 'c', 'comp', 'acc1', 'acc2']]
    df = df[df['c'] == columns]
    
    df = df.sort_values(by=['c', 'r', 'comp'])
    
    fig, ax = plt.subplots()

    # Plot each group
    for name, group in df.groupby('comp'):
        ax.plot(group['r'], group[f'acc{method}'], marker='', label=f'{name}')

    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Accuracy (Lower Better)')
    ax.set_title(f'Full vs Randomized Accuracy ({columns})')
    ax.legend(title='Components/Columns')

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
