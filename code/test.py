import time
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.decomposition import PCA

s = time.time()
X = np.array(np.random.rand(10000,10000))
pca = PCA(n_components=20)
pca.fit(X)
e = time.time()

print(e-s)
