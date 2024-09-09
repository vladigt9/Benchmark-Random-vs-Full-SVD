import numpy as np
import random
from incr_full_rand_bench.benchmark import run_benchmark_increment
from plots.increment_bench_plots import heatmap, increment_3d_accuracy_plot, increment_3d_timing_plot
from plots.rand_svd_bench_plots import rand_svd_3d_accuracy_plot, rand_svd_3d_timing_plot
from plots.sparse_bench_plots import plot_timing_sparse, sparse_3d_acc_plot
from rand_svd_bench.benchmark import run_benchmark_rand_svd
from other.merge_txt import merge_txt_files, read_csv
from rand_vs_arp_bench.benchmark import perform_rand_arp_bench
from slow_decay_acc_bench.incr_slow_acc_bench import run_benchmark_increment_slow
from slow_decay_acc_bench.solo_slow_acc_bench import run_benchmark_solo_slow

random.seed(1)
np.random.seed(1)

# Merge txt files from prem folders
# merge_txt_files(folder_path='prem_data_rand_8', output_file='merged_output_random_8.txt')

# Run increment test
# run_benchmark_increment()

# Run Single matrix randomized svd benchmarking
# run_benchmark_rand_svd()

# Run increment slow decay acc testing
# run_benchmark_increment_slow()

# Run rand svd slow decay acc testing
# run_benchmark_solo_slow()

# Increment benchmarking 3d plot of timing
# for i in range(1,11):
    # heatmap('data/final_data_incr_slow.csv', columns=i*100)
# increment_3d_timing_plot(filename='data/final_data_small_8.csv')

# Increment benchmarking plot of accuaracy
# for i in range(1,11):
#     increment_3d_accuracy_plot('slow_sm.csv', columns=i*100, method=2)
# for i in range(1,11):
    # increment_3d_accuracy_plot('data/final_data_incr_slow.csv', columns=i*100, method=2)

# Rand SVD 3d plot of timing
# rand_svd_3d_timing_plot(filename='data/final_data_1_solo.csv')


# Rnad SVD 3d plot for accuracy
# rand_svd_3d_accuracy_plot(filename_full='data/final_data_incr_slow.csv', 
#                           filename_rand='data/final_data_solo_slow.csv',
#                           normalizers=3, acc_method=2, log=False)

# perform_rand_arp_bench()

# plot_timing_sparse('test.csv', 60000)
# plot_timing_sparse('data/final_data_sparse_1.csv', 60000)
sparse_3d_acc_plot('data/final_data_sparse_8.csv', 2, False, 6000)

# merge_txt_files('prem_data_slow_sm', 'slow_sm.csv')
# read_csv('slow_sm.csv')
