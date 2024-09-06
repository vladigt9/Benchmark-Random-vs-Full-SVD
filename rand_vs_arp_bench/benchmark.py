import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
import time
import random

def create_sparse_matrix(shape, density, random_state=None):
    random.seed(random_state)
    np.random.seed(random_state)

    # Total number of elements
    total_elements = shape[0] * shape[1]
    
    # Number of non-zero elements
    num_nonzeros = int(total_elements * density)

    # Randomly choose positions for the non-zero elements
    row_indices = np.random.randint(0, shape[0], size=num_nonzeros)
    col_indices = np.random.randint(0, shape[1], size=num_nonzeros)

    # Generate random values for these positions
    data = np.random.random(size=num_nonzeros)

    # Create the sparse matrix in COO format
    sparse_matrix = coo_matrix((data, (row_indices, col_indices)), shape=shape)

    return sparse_matrix

def bench_loop_rand(sm, r:int, c:int, comp:int, d:float, n:int):
    timing_pr = []
    singular_values = np.zeros(shape=(n, comp))
    
    for i in range(n):
        pr = TruncatedSVD(n_components=comp, algorithm='randomized', random_state=1)

        s1 = time.time()
        pr.fit(sm)
        e1 = time.time()
        
        timing_pr.append(e1-s1)
        
        singular_values[i] = np.array(pr.singular_values_)
    
    with open(f'prem_data_final_test/rand_{r}_{c}_{comp}_{d}.txt', 'w') as file:
        file.write(','.join(map(str, ['row', 'randomized', r, c, sum(timing_pr)/n,
                                      comp, d,
                                      '; '.join(map(str, list(np.mean(singular_values, axis=0))))])))
        
def bench_loop_arp(sm, r:int, c:int, comp:int, d:float, n:int):
    timing_pa = []
    singular_values = np.zeros(shape=(n, comp))
    
    for i in range(n):
        pa = TruncatedSVD(n_components=comp, algorithm='arpack', random_state=1)

        s2 = time.time()
        pa.fit(sm)
        e2 = time.time()
        
        timing_pa.append(e2-s2)
        
        singular_values[i] = np.array(pa.singular_values_)
    
    with open(f'prem_data_final_test/arp_{r}_{c}_{comp}_{d}.txt', 'w') as file:
        file.write(','.join(map(str, ['row', 'arpack', r, c, sum(timing_pa)/n,
                                      comp, d,
                                      '; '.join(map(str, list(np.mean(singular_values, axis=0))))])))

def perform_rand_arp_bench():
    
    rows = [40000, 50000, 60000]
    cols = [2000, 4000, 6000]
    components = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    density = [0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    for r in rows:
        print(r)
        for c in cols:
            print(c)
            for d in density:
                print(d)
                sparse_matrix = create_sparse_matrix((r, c), density=d, random_state=1)
                for comp in components:
                    print(comp)
                    bench_loop_arp(sparse_matrix, r, c, comp, d, 10)
                    bench_loop_rand(sparse_matrix, r, c, comp, d, 10)
