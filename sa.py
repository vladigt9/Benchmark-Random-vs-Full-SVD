import numpy as np
import random
import time
from sklearn.utils.extmath import randomized_svd

random.seed(1)
np.random.seed(1)
    
X = np.array(np.random.rand(1000,100))

U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
n = (([100]*(int(len(Sigma)*2/100)))+([0.01]*(int(len(Sigma)*98/100))))
X = U@(np.diag(Sigma)*n)@Vt
    
# Lists created to store test data
times_rand = []
sing_values = []
    
for i in range(2):
    # if i ==0:
    #     print(i)
    # Get time need for PCA transformation
    s = time.time()
    U, s, Vh = randomized_svd(X, n_components=10, random_state=1)
    e = time.time()
    times_rand.append(e-s)
    
    sing_values=s
    
    print(s)
