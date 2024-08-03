import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time

def test(rows:int, cols:int, n_simulations:int, n_components:int, solver:str) -> float:
    
    # Multipliers for changing the covariance of the original data set: X2 = U*Sigma*n*V'
    multipliers = [0.2, 0.5, 1, 2, 5]
    
    for m in multipliers:
    
        # Reset random seed to always get the same matrix
        random.seed(1)
        np.random.seed(1)
        
        X = np.array(np.random.rand(rows,cols))
    
        # Scale half of the singluar values of the original matrix to create a new one
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
        n = (([m]*(int(len(Sigma)/2)))+([1]*(int(len(Sigma)/2))))
        X = U@(np.diag(Sigma)*n)@Vt
        
        # Lists created to store test data
        times_rand = []
        expl_var = []
        clust_dist = []
        sing_values = []
        
        for i in range(n_simulations):

            # Get time need for PCA transformation
            pca = PCA(n_components=n_components, svd_solver=solver, random_state=1)
            s = time.time()
            pca.fit(X)
            e = time.time()
            times_rand.append(e-s)
            
            # get the explained variance and singular values on the first loop only
            # as they are the same each time
            if i == 0:
                expl_var.append(pca.explained_variance_ratio_[:10])
                sing_values = pca.singular_values_

            # Transform the pca data so that clustering can be performed
            pca_data = pca.transform(X)
            
            kmeans = KMeans(n_clusters=3, random_state=1)
            kmeans.fit(pca_data)
            
            # calculate average distance to cluster senter
            distances = np.linalg.norm(pca_data - kmeans.cluster_centers_[kmeans.labels_], axis=1)
            clust_dist.append(np.mean(distances))
            
    # write a file saving the extracted data from simulations
    with open(f'prem_data/{rows}{solver}{solver}.txt', 'w') as file:
        file.write(','.join(map(str, [solver, rows, cols, sum(times_rand)/n_simulations, 
                                '; '.join(map(str, expl_var)), sum(clust_dist)/n_simulations,
                                '; '.join(map(str, sing_values))])))
        
    return 

def solvers_timing() -> None:
    # Create lists of the solvers and list of % used to set the number of columns
    solvers = ['randomized', 'full', 'covariance_eigh']
    cols_p = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    # set the initial number of rows of the matrix
    i = 50000
    
    # set the cap for number of columns
    while i <= 150000:
        
        for k in cols_p:
            # print used to keep track of simulation
            print(i,k)
            # set
            for s in solvers: 
                test(i,int(i*k),50,20,s)
        
        i+=10000

    return
