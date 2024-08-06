import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from sklearn.utils._array_api import get_namespace
import random
import time

random.seed(1)
np.random.seed(1)

X = np.array(np.random.rand(50000,1000))
U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
n = (([100]*(int(len(Sigma)*2/100)))+([0.01]*(int(len(Sigma)*98/100))))
X = U@(np.diag(Sigma)*n)@Vt

xp, _ = get_namespace(X)
mean_ = xp.mean(X, axis=0)
X_centered = xp.asarray(X, copy=True)
X_centered -= mean_
U, s, Vh = randomized_svd(X_centered, n_components=100, random_state=1,
                                  n_oversamples=10, n_iter=2,
                                  power_iteration_normalizer='LU', svd_lapack_driver='gesdd')
U2, s2, Vh2 = randomized_svd(X_centered, n_components=100, random_state=1,
                                  n_oversamples=10, n_iter=2,
                                  power_iteration_normalizer='QR', svd_lapack_driver='gesdd')

sing_values1 = s
sing_values2 = s2  

# print(e1-s1)
# print(e2-s2)
print(np.linalg.norm((sing_values1 - sing_values2)/sing_values1))
print(np.linalg.norm(sing_values1 - sing_values2)/np.linalg.norm(sing_values1))
