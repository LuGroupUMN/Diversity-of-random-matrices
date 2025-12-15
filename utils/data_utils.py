# matrix A for the linear system Ax=b where A represents a finite-element discretization of the operator
# u(x) -> -∆ (a(x)∇u(x)) + V(x)u(x)
# u(0) = u(1) = 0
import numpy as np
from numpy import kron
from numpy.polynomial.legendre import leggauss
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import torch
from tqdm import tqdm
from math import sqrt

def get_noise(N, n, d, device='cpu'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    x = torch.randn(N, n, d, device=device)
    Yn = (x.transpose(1, 2) @ x) / n
    return Yn

def get_noise_batch(N, n, d, batch_size=32, device='cpu'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')    
    results = []
    for i in range(0, N, batch_size):
        curr_batch = min(batch_size, N - i)
        Yn_batch = get_noise(curr_batch, n, d, device=device)
        results.append(Yn_batch)
        del Yn_batch
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return torch.cat(results, dim=0)

def get_a(atype, alpha=2):
    if atype == 'const':
        return lambda x: 1.0 + 0*x
    if atype == 'const2':
        return lambda x: 0.1 + 0*x
    if atype == 'const3':
        return lambda x: 5 + 0*x
    if atype == 'lognormal':
        num_terms = 100
        xi = np.random.randn(num_terms, 1)
        k = np.arange(1, num_terms + 1).reshape(-1, 1)
        lambdas = 1 / (np.pi * np.pi * k * k)
        
        def lognormal_field(x):
            x = np.atleast_1d(x)
            sin_terms = np.sin(k * np.pi * x.reshape(1, -1))
            terms = np.sqrt(lambdas) * xi * sin_terms
            return np.exp(np.sum(terms, axis=0))
        
        return lognormal_field

def get_a_2d(atype, alpha=2):
    if atype == 'const':
        return lambda x, y: 1.0 + 0*x*y
    if atype == 'const2':
        return lambda x, y: 0.1 + 0*x*y
    if atype == 'const3':
        return lambda x, y: 5 + 0*x*y
    if atype == 'lognormal':
        f = get_a('lognormal', alpha=alpha)
        g = get_a('lognormal', alpha=alpha)
        return lambda x, y: f(x) * g(y)

def get_V(Vtype, d, alpha=2):
    if Vtype == 'piecewise':
        values = np.random.choice([1, 2], size=d)
        
        def piecewise_func(x):
            x = np.atleast_1d(x)
            bins = np.linspace(0, 1, d + 1)
            indices = np.digitize(x, bins) - 1
            indices = np.clip(indices, 0, d - 1)
            return values[indices].astype(float)
        
        return piecewise_func
    if Vtype == 'lognormal':
        num_terms = 100
        xi = np.random.randn(num_terms, 1)
        k = np.arange(1, num_terms + 1).reshape(-1, 1)
        lambdas = 1 / (np.pi * np.pi * k * k)
        
        def lognormal_field(x):
            x = np.atleast_1d(x)
            sin_terms = np.sin(k * np.pi * x.reshape(1, -1))
            terms = np.sqrt(lambdas) * xi * sin_terms
            return np.exp(np.sum(terms, axis=0))
        
        return lognormal_field

def get_V_2d(Vtype, d, alpha=2):
    if Vtype == 'piecewise':
        f = get_V('piecewise', d=d)
        g = get_V('piecewise', d=d)
        return lambda x, y: f(x) * g(y)
    if Vtype == 'lognormal':
        f = get_V('lognormal', d=d, alpha=alpha)
        g = get_V('lognormal', d=d, alpha=alpha)
        return lambda x, y: f(x) * g(y)


def generate_A_1d_fd(d, a, V):
    A = np.zeros((d, d))
    h = 1 / (d + 1)

    A[0, 0] = a((0.5) * h) + a((1.5) * h) + h**2 * V(1 * h)
    A[0, 1] = -a((1.5) * h)

    for i in range(1, d-1):
        A[i, i-1] = -a((i + 0.5) * h)
        A[i, i]   = a((i + 0.5) * h) + a((i + 1.5) * h) + h**2 * V((i + 1) * h)
        A[i, i+1] = -a((i + 1.5) * h)

    A[-1, -2] = -a((d - 0.5) * h)
    A[-1, -1] = a((d - 0.5) * h) + a((d + 0.5) * h) + h**2 * V(d * h)

    return A / (h**2)

def generate_A_1d_fd_batch(N, d, a, V):
    res = np.zeros((N, d, d))
    for i in tqdm(range(N)):
        res[i, :, :] = generate_A_1d_fd(d, a[i], V[i])
    return res

def generate_A_1d_fem_OLD(d, a, V, num_quad=256):
    # Local helper functions from your snippet
    def phi(k, x):
        if k < 0: return None
        result = np.zeros_like(x)
        if k == 0:
            mask = (0 <= x) & (x <= 1/(d-1))
            result[mask] = (d-1) * (1/(d-1) - x[mask])
        elif k == d-1:
            mask = ((d-2)/(d-1) <= x) & (x <= 1)
            result[mask] = (d-1) * (x[mask] - (d-2)/(d-1))
        else:
            mask1 = ((k-1)/(d-1) <= x) & (x <= k/(d-1))
            result[mask1] = (d-1) * (x[mask1] - (k-1)/(d-1))
            mask2 = (k/(d-1) < x) & (x <= (k+1)/(d-1))
            result[mask2] = (d-1) * ((k+1)/(d-1) - x[mask2])
        return result
    
    def dphi(k, x):
        if k < 0: return None
        result = np.zeros_like(x)
        if k == 0:
            mask = (0 <= x) & (x <= 1/(d-1))
            result[mask] = -(d-1)
        elif k == d-1:
            mask = ((d-2)/(d-1) <= x) & (x <= 1)
            result[mask] = (d-1)
        else:
            mask1 = ((k-1)/(d-1) <= x) & (x <= k/(d-1))
            result[mask1] = (d-1)
            mask2 = (k/(d-1) < x) & (x <= (k+1)/(d-1))
            result[mask2] = -(d-1)
        return result
    
    def integrate_01(f):
        x, w = leggauss(num_quad)
        x = 0.5 * (x + 1)
        w = 0.5 * w
        return np.sum(w * f(x))
    
    A = np.zeros((d, d))
    for j in range(d):
        for k in range(d):
            A[j, k] = (
                integrate_01(lambda x: a(x) * dphi(j, x) * dphi(k, x)) +
                integrate_01(lambda x: phi(j, x) * phi(k, x) * V(x))
            )
    return A

def generate_A_1d_fem(d, a, V, num_quad=16): 
    h = 1.0 / (d - 1)
    
    def integrate_interval(func, x_start, x_end):
        # integrate func between x_start, x_end
        xi, w = leggauss(num_quad)

        # map to [x_start, x_end]
        mid = 0.5 * (x_start + x_end)
        scale = 0.5 * (x_end - x_start)
        x_mapped = mid + scale * xi
        w_mapped = w * scale
        return np.sum(w_mapped * func(x_mapped))

    def phi(k, x):
        if k < 0: return None
        result = np.zeros_like(x)
        if k == 0:
            mask = (0 <= x) & (x <= 1/(d-1))
            result[mask] = (d-1) * (1/(d-1) - x[mask])
        elif k == d-1:
            mask = ((d-2)/(d-1) <= x) & (x <= 1)
            result[mask] = (d-1) * (x[mask] - (d-2)/(d-1))
        else:
            mask1 = ((k-1)/(d-1) <= x) & (x <= k/(d-1))
            result[mask1] = (d-1) * (x[mask1] - (k-1)/(d-1))
            mask2 = (k/(d-1) < x) & (x <= (k+1)/(d-1))
            result[mask2] = (d-1) * ((k+1)/(d-1) - x[mask2])
        return result
    
    def dphi(k, x):
        if k < 0: return None
        result = np.zeros_like(x)
        if k == 0:
            mask = (0 <= x) & (x <= 1/(d-1))
            result[mask] = -(d-1)
        elif k == d-1:
            mask = ((d-2)/(d-1) <= x) & (x <= 1)
            result[mask] = (d-1)
        else:
            mask1 = ((k-1)/(d-1) <= x) & (x <= k/(d-1))
            result[mask1] = (d-1)
            mask2 = (k/(d-1) < x) & (x <= (k+1)/(d-1))
            result[mask2] = -(d-1)
        return result

    A = np.zeros((d, d))
    for j in range(d):
        start_k = max(0, j - 1)
        end_k = min(d, j + 2)
        
        for k in range(start_k, end_k):
            
            # integration limits
            x_j = j * h
            x_k = k * h
            
            # The intersection of support for phi_j and phi_k
            lower_bound = max(0, min(x_j, x_k) - h)
            upper_bound = min(1, max(x_j, x_k) + h)
            
            
            val = 0.0
            
            x_left = min(x_j, x_k)            
            limit_a = max(j-1, k-1) * h
            limit_b = min(j+1, k+1) * h
            def integrand(x):
                 return a(x) * dphi(j, x) * dphi(k, x) + phi(j, x) * phi(k, x) * V(x)
            if limit_a < 0: limit_a = 0
            if limit_b > 1: limit_b = 1
            
            split_node = max(j, k) * h
            
            if limit_a < split_node < limit_b:
                val += integrate_interval(integrand, limit_a, split_node)
                val += integrate_interval(integrand, split_node, limit_b)
            else:
                val += integrate_interval(integrand, limit_a, limit_b)

            A[j, k] = val
            
    return A

def generate_A_1d_fem_batch(N, d, a, V, num_quad=16):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # a_t = torch.tensor(a, device=device)
    # V_t = torch.tensor(V, device=device)
    res = torch.zeros((N, d, d), device=device)
    
    for i in tqdm(range(N)):
        res[i] = torch.from_numpy(generate_A_1d_fem(d, a[i], V[i], num_quad)).float().to(device)
    
    return res.cpu().numpy()

def generate_A_2d_fd(d, a, V):
    if int(sqrt(d))**2 != d:
        raise ValueError(f'For 2D generation, d={d} must be a perfect square')
    d = int(sqrt(d))

    A = lil_matrix((d*d, d*d)) 
    h = 1 / (d + 1)

    for i in range(d):
        for j in range(d):
            row = i * d + j
            x = (i + 1) * h
            y = (j + 1) * h

            # Coefficients at half-steps
            a_right = a(x + 0.5 * h, y)
            a_left  = a(x - 0.5 * h, y)
            a_up    = a(x, y + 0.5 * h)
            a_down  = a(x, y - 0.5 * h)

            # Diagonal element
            diag = (a_right + a_left + a_up + a_down) + (h**2 * V(x, y))
            A[row, row] = diag

            # Off-diagonal elements (Neighbors)
            if i + 1 < d: A[row, (i + 1) * d + j] = -a_right
            if i - 1 >= 0: A[row, (i - 1) * d + j] = -a_left
            if j + 1 < d: A[row, i * d + (j + 1)] = -a_up
            if j - 1 >= 0: A[row, i * d + (j - 1)] = -a_down

    return A.toarray() / (h**2)

def generate_A_2d_fd_batch(N, d, a, V, device = None):
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # a_t = torch.tensor(a, device=device)
    # V_t = torch.tensor(V, device=device)
    res = torch.zeros((N, d, d), device=device)
    
    for i in tqdm(range(N)):
        res[i] = torch.from_numpy(generate_A_2d_fd(d, a[i], V[i])).float().to(device)
    
    return res.cpu().numpy()

def generate_A_2d_fem(d, a, V, num_quad=16):
    if int(sqrt(d))**2 != d:
        raise ValueError(f'For 2D generation, d={d} must be a perfect square')
    d = int(sqrt(d))

    def integrate_01(f):
        x, w = leggauss(num_quad)
        x = 0.5 * (x + 1)
        w = 0.5 * w
        return np.sum(w * f(x))

    def phi(k, x):
        if k < 0: return None
        result = np.zeros_like(x)
        if k == 0:
            mask = (0 <= x) & (x <= 1/(d-1))
            result[mask] = (d-1) * (1/(d-1) - x[mask])
        elif k == d-1:
            mask = ((d-2)/(d-1) <= x) & (x <= 1)
            result[mask] = (d-1) * (x[mask] - (d-2)/(d-1))
        else:
            mask1 = ((k-1)/(d-1) <= x) & (x <= k/(d-1))
            result[mask1] = (d-1) * (x[mask1] - (k-1)/(d-1))
            mask2 = (k/(d-1) < x) & (x <= (k+1)/(d-1))
            result[mask2] = (d-1) * ((k+1)/(d-1) - x[mask2])
        return result
    
    def dphi(k, x):
        if k < 0: return None
        result = np.zeros_like(x)
        if k == 0:
            mask = (0 <= x) & (x <= 1/(d-1))
            result[mask] = -(d-1)
        elif k == d-1:
            mask = ((d-2)/(d-1) <= x) & (x <= 1)
            result[mask] = (d-1)
        else:
            mask1 = ((k-1)/(d-1) <= x) & (x <= k/(d-1))
            result[mask1] = (d-1)
            mask2 = (k/(d-1) < x) & (x <= (k+1)/(d-1))
            result[mask2] = -(d-1)
        return result

    a1 = lambda x: a(x, 1)
    a2 = lambda y: a(1, y) / a(1, 1)
    V1 = lambda x: V(x, 1)
    V2 = lambda y: V(1, y) / V(1, 1)

    # Need to build: S(a1), S(a2), M(a1), M(a2), M(V1), M(V2)
    Sa1 = np.zeros((d, d))
    Sa2 = np.zeros((d, d))
    Ma1 = np.zeros((d, d))
    Ma2 = np.zeros((d, d))
    MV1 = np.zeros((d, d))
    MV2 = np.zeros((d, d))
    # A_a = S(a1) X M(a2) + M(a1) X S(a2)

    # Build S(a1)
    for j in range(d):
        for k in range(d):
            Sa1[j, k] = integrate_01( lambda x: a1(x) * dphi(j, x) * dphi(k, x) )
            Sa2[j, k] = integrate_01( lambda x: a2(x) * dphi(j, x) * dphi(k, x) )
            Ma1[j, k] = integrate_01( lambda x: a1(x) * phi(j, x) * phi(k, x) )
            Ma2[j, k] = integrate_01( lambda x: a2(x) * phi(j, x) * phi(k, x) )
            MV1[j, k] = integrate_01( lambda x: V1(x) * phi(j, x) * phi(k, x) )
            MV2[j, k] = integrate_01( lambda x: V2(x) * phi(j, x) * phi(k, x) )
    
    A = kron(Sa1, Ma2) + kron(Ma1, Sa2) + kron(MV1, MV2)
    return A

def generate_A_2d_fem_batch(N, d, a, V, num_quad=16):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # a_t = torch.tensor(a, device=device)
    # V_t = torch.tensor(V, device=device)
    res = torch.zeros((N, d, d), device=device)
    
    for i in tqdm(range(N)):
        res[i] = torch.from_numpy(generate_A_2d_fem(d, a[i], V[i], num_quad)).float().to(device)
    
    return res.cpu().numpy()