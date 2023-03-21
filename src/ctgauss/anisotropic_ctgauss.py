# Copyright 2023 Jesse Windle                                                                                     

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file.


import numpy as np
import scipy as sp
from ctgauss.ctgauss_base import CTGaussBase


class AnisotropicCTGauss(CTGaussBase):

    def __init__(self, M, mu_or_r, A, y, F, g, L, mean=True):
        super().__init__(A, y, F, g, L)
        self.M = M
        if mean:
            self.r = np.matmul(M, mu_or_r)
        else:
            self.r = mu_or_r


    def get_ode_params(self, j):
        n, d = self.A[j].shape
        Q, R = np.linalg.qr(self.A[j], mode='complete')
        Q1 = Q[:,0:d]
        R1 = R[0:d,:]
        Q2 = Q[:,d: ]
        Om22 = np.matmul(np.matmul(Q2.T, self.M), Q2)
        U = sp.linalg.cholesky(Om22, lower=False)
        Sp = sp.linalg.solve_triangular(U, Q2.T, trans=1, lower=False)
        z1 = sp.linalg.solve_triangular(R1, -self.y[j], trans=1, lower=False) # A'x + y
        x1 = np.matmul(Q1, z1)
        rtil = self.r - np.matmul(self.M, x1)
        offset = np.matmul(Sp.T, np.matmul(Sp, rtil))
        xp = np.matmul(Sp.T, np.matmul(Sp, rtil)) + x1
        S = np.transpose(Sp)
        return (xp, Q, S)


    def get_potential(self, j, x):
        return 0.5 * np.dot(np.matmul(self.M, x), x) - np.dot(self.r, x)
