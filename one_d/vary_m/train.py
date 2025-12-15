"""
x = N(0, I)
y = Ainv @ x
# so, Ay = x
# we operate on y, and get x

Ainvn = Ainv @ noise
train_data = [x, y, Ainv, Ainvn]
train_model(Ainv, Ainvn, x, y)

def train_model(Ainv, Ainvn, x, y):
	ypred = model(Ainvn, x)
	loss = MSELoss(ypred, ytrain)
	# we adjust params, trying to get model(Ainvn, x) as close as possible to Ainv @ x"""

import torch
import numpy
from .config import *
from utils.train_utils import train_model
from utils.data_utils import generate_A_1d_fd_batch, get_noise_batch, get_a, get_V

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

print(f'(1D) Generating train matrix A of size {N}x{d}x{d}')
alist = np.array([get_a(atype, alpha) for _ in range(N)])    # N x (numpy function)
Vlist = np.array([get_V(Vtype, d, alpha) for _ in range(N)]) # N x (numpy function)
A = generate_A_1d_fd_batch(N, d, alist, Vlist) # N x d x d
A = torch.from_numpy(A).float().to(device)
Ainv = torch.inverse(A)

x = torch.randn(N, d, device=device) # N x d
y = Ainv @ x.unsqueeze(2)
y = torch.squeeze(y)

noise = get_noise_batch(N, n, d, device=device)
# noise = torch.from_numpy(noise).float().to(device)
Ainvn = Ainv @ noise

mdl = train_model(Ainvn, x, y, d, lr=lr, num_epochs=num_epochs)
torch.save(mdl, 
	f"trained_model"
)