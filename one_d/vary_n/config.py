import numpy as np
d = 32
N = 30000
nlist = np.array([512, 1024, 2048, 4096, 8192])
M = 1000
m = 80000
alpha = 3
atype = 'const2'
Vtype = 'piecewise'

lr = 0.0005
num_epochs = 5000

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