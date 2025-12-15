import torch
import numpy
from .config import *
from utils.train_utils import eval_model
from utils.data_utils import generate_A_2d_fem_batch, get_a_2d, get_V_2d, get_noise

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.bakends.mps.is_available() else 'cpu')

mdl = torch.load("trained_model", weights_only=False, map_location=device)

print(f'(2D) Generating test matrix A of size {M}x{d}x{d}')
alist = np.array([get_a_2d(atype, alpha) for _ in range(M)])    # M x (numpy function)
Vlist = np.array([get_V_2d(Vtype, d, alpha) for _ in range(M)]) # M x (numpy function)
A = generate_A_2d_fem_batch(M, d, alist, Vlist) # M x d x d
A = torch.from_numpy(A).float().to(device)
Ainv = torch.inverse(A)

x = torch.randn(M, d, device=device) # M x d
y = Ainv @ x.unsqueeze(2)
y = torch.squeeze(y)

print(f'm    | L2')
for m in mlist:    
    noise = get_noise(M, m, d, device)
    # noise = torch.from_numpy(noise).float().to(device)
    Ainvm = Ainv @ noise
    
    l2 = eval_model(mdl, Ainvm, x, y)
    print(f'{int(m):<5}', l2)