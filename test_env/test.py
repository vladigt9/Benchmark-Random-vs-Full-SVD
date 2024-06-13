import random
import time

def test_f():
    random.seed(1)
    times = time.time()
    for i in range(1000):
        for k in range(10000):
                z = random.randrange(0,10)
    e = time.time()

    print(z)
    print(times-e)    
