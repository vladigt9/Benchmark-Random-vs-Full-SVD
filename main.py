import pandas as pd
import numpy as np
import random
from genuineVsRand.data import createHiggsData, createHconsData
from genuineVsRand.tests import time_test

random.seed(1)
np.random.seed(1)


def completeTests():
    # df_hi = createHiggsData()
    df_hc = createHconsData()
    df_hc = df_hc.iloc[:1000000, :]
    
    i = 100000
    while i <= 500000:

        for k in range(50,51, 10):
            time_test(df_hc, i, k, 100, 5, 'covariance_eigh')
        
        i+=100000

completeTests()
