import numpy as np
import random
from incr_full_rand_bench.benchmark import run_benchmark_increment
from plots.increment_bench_plots import increment_3d_accuracy_plot, increment_3d_timing_plot
from plots.rand_svd_bench_plots import rand_svd_3d_accuracy_plot, rand_svd_3d_timing_plot
from rand_svd_bench.benchmark import run_benchmark_rand_svd
from other.merge_txt import merge_txt_files
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
# increment_3d_timing_plot(filename='data/final_data_8_fvr.csv')

# Increment benchmarking 3d plot of accuaracy
# increment_3d_accuracy_plot('data/final_data_incr_slow.csv', columns=100, method=1)
# increment_3d_accuracy_plot('data/final_data_8_fvr.csv', columns=100, method=1)

# Rand SVD 3d plot of timing
# rand_svd_3d_timing_plot(filename='data/final_data_8_solo.csv')


# Rnad SVD 3d plot for accuracy
rand_svd_3d_accuracy_plot(filename_full='data/final_data_incr_slow.csv', 
                          filename_rand='data/final_data_solo_slow.csv',
                          normalizers=2, acc_method=2, log=False)
