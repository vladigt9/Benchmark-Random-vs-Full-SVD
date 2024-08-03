import numpy as np
import random
import time
from sklearn.decomposition import PCA

def test(rows:int, cols:int, n_simulations:int, n_components:int, solver:str) -> float:
    
    # Multipliers for changing the covariance of the original data set: X2 = U*Sigma*n*V'
    # multipliers = [0.2,1,5]
    
    # for m in multipliers:
        # print(m)
    
        # Reset random seed to always get the same matrix
    random.seed(1)
    np.random.seed(1)
    
    X = np.array(np.random.rand(rows,cols))

    # Scale half of the singluar values of the original matrix to create a new one
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    n = (([100]*(int(len(Sigma)*2/100)))+([0.01]*(int(len(Sigma)*98/100))))
    X = U@(np.diag(Sigma)*n)@Vt
    
    # Lists created to store test data
    times_rand = []
    expl_var = []
    sing_values = []
    
    for i in range(n_simulations):
        # if i ==0:
        #     print(i)
        # Get time need for PCA transformation
        pca = PCA(n_components=int(cols*n_components), svd_solver=solver, random_state=1)
        s = time.time()
        pca.fit(X)
        e = time.time()
        times_rand.append(e-s)
        
        # get the explained variance and singular values on the first loop only
        # as they are the same each time
        if i == 0:
            expl_var.append(pca.explained_variance_ratio_[:10])
            sing_values = pca.singular_values_
        
    # write a file saving the extracted data from simulations
    with open(f'prem_data/{rows}_{cols}_intelex_{n_components}.txt', 'w') as file:
        file.write(','.join(map(str, [solver, rows, cols, n_components, sum(times_rand)/n_simulations, 
                                '; '.join(map(str, expl_var)),
                                '; '.join(map(str, sing_values))])))
    
    return 

def solvers_timing() -> None:
    # Create lists of the solvers and list of % used to set the number of columns
    # solvers = ['randomized', 'full', 'covariance_eigh']
    solvers = ['full']
    # cols_p = [0.001, 0.005, 0.01]
    cols_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # components = [10,20,30,40,50,60,70,80,90]
    components = [0.1, 0.2,0.3,0.4,0.5,0.6]
    # set the initial number of rows of the matrix
    i = 90000
    
    
    # set the cap for number of columns
    while i <= 105000:
        
        for comp in components:
            print(f'comp {comp}')
            for k in cols_p:
                # print used to keep track of simulation
                print(i,k)
                # set
                for s in solvers:
                    print('intel') 
                    test(i,k,20,comp,s)
        
        i+=10000

    return


# region is between 0.3 and 0.6
