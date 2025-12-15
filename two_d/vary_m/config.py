import numpy as np
d = 16  # must be a perfect square
N = 20000
n = 40000
M = 1000
mlist = np.array([512, 1024, 2048, 4096, 8192, 16384])
alpha = 2
atype = 'lognormal'
Vtype = 'lognormal'

lr = 0.001
num_epochs = 10000