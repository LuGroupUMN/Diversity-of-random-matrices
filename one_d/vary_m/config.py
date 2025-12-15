import numpy as np
d = 10
N = 40000
n = 40000
M = 1000
mlist = np.array([512, 1024, 2048, 4096, 8192, 16384])
alpha = 3
atype = 'lognormal'
Vtype = 'lognormal'

lr = 0.001
num_epochs = 10000

# config = {
# 	'd': 9,
# 	'N': 5000,
# 	'n': 10000,
# 	'M': 10000,
# 	'mlist': np.array([512, 1024, 2048, 4096, 8192, 16384]),
# 	'alpha': 2,
# 	'atype': 'const2',
# 	'Vtype': 'piecewise'
# }