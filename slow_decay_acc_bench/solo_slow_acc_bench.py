import numpy as np
import random
import time
from sklearn.utils.extmath import randomized_svd
from sklearn.utils._array_api import get_namespace

def rand_svd_benchmark_settings(X, rows:int, cols:int, n_components:int, 
         n_oversamples: int, n_iter: int, normalizer: str, driver:str) -> float:
    
    random.seed(1)
    np.random.seed(1)
    
    st = time.time()
    U, s, Vh = randomized_svd(X, n_components=int(cols*n_components), random_state=1,
                                n_oversamples=n_oversamples, n_iter=n_iter,
                                power_iteration_normalizer=normalizer, svd_lapack_driver=driver)
    e = time.time()
    times_rand = e-st
    
    sing_values = s
    
    with open(f'prem_data_solo_slow/{rows}_{cols}_{n_oversamples}_{n_iter}_{normalizer}.txt', 'w') as file:
        file.write(','.join(map(str, ['row', rows, cols, n_components, times_rand,
                                      n_oversamples, n_iter, normalizer, driver,
                                      '; '.join(map(str, sing_values))])))
    
    return 

def run_benchmark_solo_slow() -> None:
    
    cols_p = [1000]
    components = [0.1]
    n_iter = [1,2,3,4,5,6,7,8,9,10]
    n_oversamples = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    normalizer = ['QR', "LU", "none"]
    i = 50000
    
    while i <= 55000:
        
        for c in cols_p:
            print(i,c)
            random.seed(1)
            np.random.seed(1)
            X = np.array(np.random.rand(i,c))
            xp, _ = get_namespace(X)
            mean_ = xp.mean(X, axis=0)
            X_centered = xp.asarray(X, copy=True)
            X_centered -= mean_
            
            for comp in components:
                print(f'comp {comp}')
                for it in n_iter:
                    print(f'iterations: {it}')
                    for ov in n_oversamples:
                        print(f'oversamples {ov}')
                        for n in normalizer:
                            # for d in driver:
                            rand_svd_benchmark_settings(X=X_centered, rows=i,cols=c,
                                    n_components=comp,n_iter=it,
                                    n_oversamples=ov, normalizer=n, driver='gesdd')
        
        i+=10000

    return
