import numpy as np
import random
import time
from sklearn.decomposition import PCA

def increment_benchmark_settings(rows:int, cols:int, n_simulations:int, n_components:int, solver:str) -> float:
    random.seed(1)
    np.random.seed(1)
    
    X = np.array(np.random.rand(rows,cols))

    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    n = (([100]*(int(len(Sigma)*2/100)))+([0.01]*(int(len(Sigma)*98/100))))
    X = U@(np.diag(Sigma)*n)@Vt
    
    times_rand = []
    expl_var = []
    sing_values = []
    
    for i in range(n_simulations):
        pca = PCA(n_components=int(cols*n_components), svd_solver=solver, random_state=1)
        s = time.time()
        pca.fit(X)
        e = time.time()
        times_rand.append(e-s)
        
        if i == 0:
            expl_var.append(pca.explained_variance_ratio_[:10])
            sing_values = pca.singular_values_
        
    with open(f'prem_data_rand_8/{rows}_{cols}_{solver}_{n_components}.txt', 'w') as file:
        file.write(','.join(map(str, [solver, rows, cols, n_components, sum(times_rand)/n_simulations, 
                                '; '.join(map(str, expl_var)),
                                '; '.join(map(str, sing_values))])))
    
    return 

def run_benchmark_increment() -> None:
    
    solvers = ['randomized', 'full']
    cols_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    components = [0.1, 0.2,0.3,0.4,0.5,0.6]
    
    i = 90000
    
    while i <= 105000:
        
        for comp in components:
            print(f'comp {comp}')
            for k in cols_p:
                print(i,k)
                for s in solvers:
                    increment_benchmark_settings(i,k,15,comp,s)
        
        i+=10000

    return
