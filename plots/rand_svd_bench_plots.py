import pandas as pd
import matplotlib.pyplot as plt
import ast

def rand_svd_3d_timing_plot(filename):
    df = pd.read_csv(filename)
    
    df['sing_values'] = df['sing_values'].apply(ast.literal_eval)
    
    df = df[df['norm'] != 'QR']
    labels = df['norm'].unique()

    # Initialize the 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')


    # Define colors for different labels
    solver_colors = {
        labels[0]: 'blue',
        labels[1]: 'red',
        # labels[2]: 'green'
    }

    # Plot each segment with a unique color
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
                # Plot the segment
                ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])
                segment_x = []
                segment_y = []
                segment_z = []
            segment_x.append(row['n_over'])
            segment_y.append(row['n_iter'])
            segment_z.append(row['time'])
            previous_r = current_r
        
        # Plot the last segment
        if segment_x:
            ax.plot3D(segment_x, segment_y, segment_z, label=lbl, color=solver_colors[lbl])

    # Set axis labels and title
    ax.set_title('Time needed for SVD 50 000x 1 000')
    ax.set_xlabel('Oversamples')
    ax.set_ylabel('Iterations')
    ax.set_zlabel('Time needed for fitting (seconds)')

    # Add and customize legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {label: handle for handle, label in zip(handles, labels)}
    ax.legend(unique_labels.values(), unique_labels.keys())

    plt.tight_layout()
    plt.show()
