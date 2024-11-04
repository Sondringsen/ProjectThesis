import numpy as np

a = np.array([])
b = np.array([1, 2, 3])
c = np.array([4, 5, 6, 7])

print(c.flatten().shape)
print(c.shape)

a = np.concatenate([a, b])
a = np.concatenate([a, c])
print(a)