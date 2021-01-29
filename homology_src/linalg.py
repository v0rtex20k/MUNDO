import numpy as np
from scipy.linalg import eig, eigh
from typing import Dict, List, NewType, Tuple, Set
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

ndarray = NewType('numpy ndarray', np.ndarray)

def clean(M: ndarray)-> None:
    M[M < 0] = 0
    M[np.isnan(M)] = 0
    M[np.isinf(M)] = 0

def pairwise_distance_matrix(A: ndarray, metric: str='cityblock')-> ndarray:
    return squareform(pdist(A, metric=metric)) # L1 norm preferred to L2 norm here.

def compute_degree_matrix(adjacency_matrix: ndarray)-> ndarray:
    e = np.ones((adjacency_matrix.shape[0], 1))
    deg = np.matmul(adjacency_matrix, e)
    return np.diagflat(deg)

def is_symmetric(A: ndarray, rtol: float=1e-05, atol: float=1e-08)->bool:
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

def is_square(A: ndarray)-> bool:
    return True if A.shape[0] == A.shape[1] else False

def rkhs(H: ndarray)-> ndarray:
    if not is_symmetric: print('[COEMBEDDING ERROR] Cannot embed asymmetric kernel into RKHS'); exit()
    eigenvalues, eigenvectors = eig(H)
    return eigenvectors.dot(np.diag(np.sqrt(eigenvalues)))

def best_gamma_kernel(distance_matrix: ndarray)-> ndarray:
    r = None
    for i in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        rbf = laplacian_kernel(distance_matrix, gamma=i)
        n, r = rbf.shape[0], rbf.copy()
        if n*1000 <= np.count_nonzero(rbf):
            return rbf
    return r

def best_threshold(rbf: ndarray)-> ndarray:
    t = None
    for i in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        thresh = rbf.mean() - i*rbf.std()
        t_rbf = np.where((rbf < thresh) | (rbf == 1), 0, rbf)
        n, t = rbf.shape[0], t_rbf.copy()
        if n*1000 <= np.count_nonzero(t_rbf):
            return t_rbf
    return t

def make_hermitian(A: ndarray, gamma: float=None, t: float=None)-> ndarray:
    pd = pairwise_distance_matrix(A)
    rbf = laplacian_kernel(pd, gamma) if gamma else best_gamma_kernel(pd)
    t_rbf = np.where((rbf < t) | (rbf == 1), 0, rbf) if t else best_threshold(rbf)
    clean(t_rbf)
    return t_rbf

def compute_pinverse_diagonal(diag: ndarray)-> ndarray:
    m = diag.shape[0]
    i_diag = np.zeros((m,m))
    for i in range(m):
        di = diag[i, i]
        if di != 0.0:
            i_diag[i, i] = 1 / float(di)

    return i_diag

def compute_dsd_normalized(adj: ndarray, deg: ndarray, nrw: int= -1, lm: int= 1, is_normalized: bool=False)-> ndarray:
    deg_i = compute_pinverse_diagonal(deg)
    P = np.matmul(deg_i, adj)
    Identity = np.identity(adj.shape[0])
    e = np.ones((adj.shape[0], 1))

    # Compute W
    scale = np.matmul(e.T, np.matmul(deg, e))[0, 0]
    W = np.multiply(1 / scale, np.matmul(e, np.matmul(e.T, deg)))

    up_P = np.multiply(lm, P - W)
    X_ = Identity - up_P
    X_i = np.linalg.pinv(X_)

    if nrw > 0:
        LP_t = Identity - np.linalg.matrix_power(up_P, nrw)
        X_i = np.matmul(X_i, LP_t)
    
    if is_normalized == False:
        return X_i
    
    # Normalize with steady state
    SS = np.sqrt(np.matmul(deg, e))
    SS = compute_pinverse_diagonal(np.diagflat(SS))

    return np.matmul(X_i, SS)

