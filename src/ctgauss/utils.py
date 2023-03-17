# Copyright 2023 Jesse Windle                                                                                                                                                                                                       
# Use of this source code is governed by an MIT-style                                                             # license that can be found in the LICENSE file.


import numpy as np


def constraint_dimension(A, y, n):
    """Get constraint dimension and check that A and y are valid"""
    if A is None and y is None:
        return 0
    if len(A) != len(y):
        raise Exception("len(A) != len(y)")
    if len(A) == 0:
        return 0
    J = len(A)
    for j, A_j, y_j in zip(range(J), A, y):
        if not isinstance(A_j, np.ndarray):
            raise Exception(f"A[{j}] is not an ndarray")
        if not isinstance(y_j, np.ndarray):
            raise Exception(f"y[{j}] is not an ndarray")
        if len(A_j.shape) != 2:
            raise Exception(f"A[{j}] is not a matrix")
        if len(y_j.shape) > 2:
            raise Excpetion(f"y[{j}] is not a matrix or vector")
        if A_j.shape[1] != y_j.shape[0]:
            raise Exception(f"The shape of A[{j}] and y[{j}] do not agree")
        if np.any(A_j.shape != A[0].shape):
            raise Exception(f"A[{j}] is a different shape than A[0]")
        if np.any(y_j.shape != y[0].shape):
            raise Exception(f"y[{j}] is a different shape than y[0]")
        if A_j.shape[0] != n:
            raise Exception(f"A[{j}] does not have {n} rows")
        # Check rank of A_j
    return A[0].shape[1]


def resid(f, d, Q, normalize=False):
    """Residual of f projected onto the first d columns of Q
        
    Assuming d < n, we could always do Q2 Q2' f.
        
    """
    n = Q.shape[0]
    half_n = int(n/2)
    z = np.zeros(shape=f.shape)
    if (d == 0):
        x = f
    elif (d == n):
        x = z
    elif (d < half_n):
        Q1 = Q[:,0:d]
        x = f - np.matmul(Q1, np.matmul(np.transpose(Q1), f))
    elif (d >= half_n):
        Q2 = Q[:,d:]
        x = np.matmul(Q2, np.matmul(np.transpose(Q2), f))
    if normalize and not np.all(x == z):
        x = x / np.linalg.norm(x)
    return x


def minamin2d(x, y):
    if x.shape != y.shape:
        raise ValueError("x and y are expected to be the same shape")
    n = len(x)
    idc = np.array(range(n))
    xmin = np.min(x)
    idc_min = idc[x == xmin]
    sub_amin = np.argmin(y[idc_min])
    amin = idc_min[sub_amin]
    return (x[amin], y[amin], amin)
    
