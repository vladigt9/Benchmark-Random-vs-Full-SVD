import numpy as np
import random
import time
from sklearn.utils.extmath import randomized_svd

def test(rows:int, cols:int, n_simulations:int, n_components:int, 
         n_oversamples: int, n_iter: int, normalizer: str) -> float:
    
    random.seed(1)
    np.random.seed(1)
    
    X = np.array(np.random.rand(rows,cols))

    # Scale half of the singluar values of the original matrix to create a new one
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    n = (([100]*(int(len(Sigma)*2/100)))+([0.01]*(int(len(Sigma)*98/100))))
    X = U@(np.diag(Sigma)*n)@Vt
    
    # Lists created to store test data
    times_rand = []
    sing_values = []
    
    for i in range(n_simulations):
        # if i ==0:
        #     print(i)
        # Get time need for PCA transformation
        s = time.time()
        U, s, Vh = randomized_svd(X, n_components=n_components, random_state=1,
                                  n_oversamples=n_oversamples, n_iter=n_iter,
                                  power_iteration_normalizer=normalizer)
        e = time.time()
        times_rand.append(e-s)
        
        if i == 0:
            sing_values = s
        
    # write a file saving the extracted data from simulations
    with open(f'prem_data_rand/{rows}_{cols}_{n_components}_{n_oversamples}_{n_iter}_{n_oversamples}.txt', 'w') as file:
        file.write(','.join(map(str, ['row', rows, cols, n_components, sum(times_rand)/n_simulations,
                                      n_oversamples, n_iter, normalizer, 
                                      '; '.join(map(str, sing_values))])))
    
    return 

def solvers_timing() -> None:
    
    cols_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    components = [0.1, 0.2,0.3,0.4,0.5,0.6]
    n_iter = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    n_oversamples = [0,2,4,6,8,10,12,14,16,18,20]
    normalizer = ['QR', "LU", "none"]
    driver = ['gesdd', 'gesvd']
    # set the initial number of rows of the matrix
    i = 10000
    
    
    # set the cap for number of columns
    while i <= 105000:
        
        for c in cols_p:
            print(i,c)
            for comp in components:
                print(f'comp {comp}')
                for it in n_iter:
                    print(it)
                    for ov in n_oversamples:
                        print(ov)
                        for n in normalizer:
                            print(n)
                            for d in driver:
                                print(d) 
                                test(rows=i,cols=c,n_simulations=20,
                                     n_components=comp,n_iter=it,
                                     n_oversamples=ov, normalizer=n, driver=d)
        
        i+=10000

    return
