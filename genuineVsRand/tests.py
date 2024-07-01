import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time

def time_test(df, rows:int, columns:int, n_simulations:int, n_components:int, solver:str) -> None:
    times_geni = []
    times_rand = []
    df = df.iloc[:rows, :columns]
    for i in range(n_simulations):
        s1 = time.time()
        pca = PCA(n_components=n_components, svd_solver=solver)
        pca.fit(df)
        e1 = time.time()
        times_geni.append(e1-s1)

        X = np.array(np.random.rand(rows,columns))
        s2 = time.time()
        pca = PCA(n_components=n_components, svd_solver=solver)
        pca.fit(X)
        e2 = time.time()
        times_rand.append(e2-s2)
        
    print(f'Genuine data {rows}x{columns} set avr time: {sum(times_geni)/n_simulations}')
    print(f'Randomized data {rows}x{columns} set avr time: {sum(times_rand)/n_simulations}')
    return
