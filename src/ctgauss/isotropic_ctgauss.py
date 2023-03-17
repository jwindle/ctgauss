# Copyright 2023 Jesse Windle                                                                                     

# Use of this source code is governed by an MIT-style license that can
# be found in the LICENSE file.


import numpy as np
import scipy as sp
from ctgauss.ctgauss_base import CTGaussBase


class IsotropicCTGauss(CTGaussBase):

    def __init__(self, phi, mu, A, y, F, g, L):
        super().__init__(A, y, F, g, L)
        if not np.isscalar(phi) and not (np.prod(phi.shape) == 1):
            raise Exception(f"phi must be a scalar or have dimension (1,), (1,1), phi={phi}")
        if phi <= 0:
            raise Exception(f"phi (={phi}) must be positive")
        self.phi = phi
        self.mu = np.array(mu).reshape((1,)) if np.isscalar(mu) else mu # Must be array or breaks dot if mu is a scalar
        self.r = self.phi*self.mu


    def get_ode_params(self, j):
        n, d = self.A[j].shape
        Q, R = np.linalg.qr(self.A[j], mode='complete')
        Q1 = Q[:,0:d]
        R1 = R[0:d,:]
        Q2 = Q[:,d: ]
        S = Q2 / np.sqrt(self.phi)
        z1 = sp.linalg.solve_triangular(R1, self.y[j], trans=1, lower=False) # A'x - y
        x1 = np.matmul(Q1, z1)
        rtil = self.phi * (self.mu - x1)
        xp = np.matmul(S, np.matmul(S.T, rtil)) + x1
        return (xp, Q, S)


    def get_potential(self, j, x):
        if x.shape != self.r.shape:
            raise Exception(f"x={x} and mu={self.mu} have differing shapes")
        return 0.5 * self.phi * np.dot(x, x) - np.dot(self.r, x)
