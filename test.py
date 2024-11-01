import numpy as np

n = 20
m = 100_000
a = 5
r = 12.5
b = r/a

obs = np.random.gamma(a, b, size=(m, n))
mu = np.mean(obs, axis=1)
z = (mu-r)/r*np.sqrt(n*a)
filt = z >= 1.282
p = np.mean(filt)
print(p)