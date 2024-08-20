import numpy as np
import random
import time
from sklearn.decomposition import PCA

def increment_benchmark_settings(rows:int, cols:int, n_components:int, solver:str) -> float:
    random.seed(1)
    np.random.seed(1)
    
    X = np.array(np.random.rand(rows,cols))
    
    pca = PCA(n_components=int(cols*n_components), svd_solver=solver, random_state=1)
    s = time.time()
    pca.fit(X)
    e = time.time()
    times_rand = e-s
    
    expl_var = pca.explained_variance_ratio_[:10]
    sing_values = pca.singular_values_

    with open(f'prem_data/{rows}_{cols}_{solver}_{n_components}.txt', 'w') as file:
        file.write(','.join(map(str, [solver, rows, cols, n_components, times_rand, 
                                '; '.join(map(str, expl_var)),
                                '; '.join(map(str, sing_values))])))
    
    return 

def run_benchmark_increment_slow() -> None:
    
    solvers = ['randomized', 'full']
    cols_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    components = [0.1, 0.2,0.3,0.4,0.5,0.6]
    
    i = 10000
    
    while i <= 105000:
        
        for comp in components:
            print(f'comp {comp}')
            for k in cols_p:
                print(i,k)
                for s in solvers:
                    increment_benchmark_settings(i,k,comp,s)
        
        i+=10000

    return
