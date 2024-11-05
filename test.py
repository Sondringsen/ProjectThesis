import numpy as np
import pandas as pd

def f1(x):
    counter = 1

    def f2():
        nonlocal counter
        counter += 1

    for i in range(3):
        f2()
    return counter


counter = f1(1)
print(counter)