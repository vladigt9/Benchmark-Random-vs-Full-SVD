import numpy as np
from sklearn.decomposition import PCA
import random
import time

random.seed(1)
np.random.seed(1)

X = np.array(np.random.rand(10000,1000))
U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
n = (([100]*(int(len(Sigma)*2/100)))+([0.01]*(int(len(Sigma)*98/100))))
X = U@(np.diag(Sigma)*n)@Vt
pca1 = PCA(n_components=10, svd_solver='full', random_state=1)
print(0)
s1 = time.time()
pca1.fit(X)
e1 = time.time()
print(1)
pca2 = PCA(n_components=10, svd_solver='randomized', random_state=1)
s2 = time.time()
pca2.fit(X)
e2 = time.time()

sing_values1 = pca1.singular_values_
sing_values2 = pca2.singular_values_

print(np.sum(pca1.explained_variance_ratio_[:10]))
print(np.sum(pca2.explained_variance_ratio_[:10]))

# print(e1-s1)
# print(e2-s2)
print(np.linalg.norm((sing_values1 - sing_values2)/sing_values1))
