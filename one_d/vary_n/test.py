import torch
import numpy
from .config import *
from utils.train_utils import eval_model
from utils.data_utils import generate_A_1d_fd_batch, get_noise, get_a, get_V, get_noise_batch
from utils.data_utils import generate_A_2d_fem_batch, get_a_2d, get_V_2d

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.bakends.mps.is_available() else 'cpu')


print(f'(1D) Generating test matrix A of size {M}x{d}x{d}')
alist = np.array([get_a(atype, alpha) for _ in range(M)])    # N x (numpy function)
Vlist = np.array([get_V(Vtype, d, alpha) for _ in range(M)]) # N x (numpy function)
A = generate_A_1d_fem_batch(M, d, alist, Vlist) # N x d x d
A = torch.from_numpy(A).float().to(device)
Ainv = torch.inverse(A)

x = torch.randn(M, d, device=device) # N x d
y = Ainv @ x.unsqueeze(2)
y = torch.squeeze(y)

noise = get_noise_batch(M, m, d, 128, device=device)
Ainvm = Ainv @ noise

print(f'n    | L2')
for n in nlist:
    mdl = torch.load(f"trained_model_{n}", weights_only=False, map_location=device)
    
    l2 = eval_model(mdl, Ainvm, x, y)
    print(f'{int(n):<5}', l2)