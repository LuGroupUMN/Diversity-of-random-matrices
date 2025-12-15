import torch
import numpy
from .config import *
from utils.train_utils import train_model
from utils.data_utils import generate_A_2d_fd_batch, get_a_2d, get_V_2d, get_noise, get_noise_batch
print(atype, Vtype)

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.bakends.mps.is_available() else 'cpu')

print(f'(2D) Generating train matrix A of size {N}x{d}x{d}')
alist = np.array([get_a_2d(atype, alpha) for _ in range(N)])    # N x (numpy function)
Vlist = np.array([get_V_2d(Vtype, d, alpha) for _ in range(N)]) # N x (numpy function)
# print(alist[0](0.1, 0.1), alist[1](0.1, 0.1))
# print(Vlist[0](0.1, 0.1), Vlist[1](0.1, 0.1))
A = generate_A_2d_fd_batch(N, d, alist, Vlist) # N x d x d
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