import numpy as np
import pandas as pd

A = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
B = np.pad(A, 2)
print(A)
print(B)