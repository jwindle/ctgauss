# Copyright 2023 Jesse Windle

# Use of this source code is governed by an MIT-style license that can
# be found in the LICENSE file.


import numpy as np
import pytest


def test_dimension(isotropic_halfspace_sym_reg):
    ictg, phimu, param = isotropic_halfspace_sym_reg
    assert ictg.dim == 3
    assert ictg.dim_reduction == 1


def test_continuity(isotropic_halfspace_sym_reg):
    ictg, phimu, param = isotropic_halfspace_sym_reg
    error_array = ictg.continuity_error()
    assert np.linalg.norm(error_array) < 1e-12


def test_sample_runs(isotropic_halfspace_sym_reg, rng):
    ictg, phimu, param = isotropic_halfspace_sym_reg
    N = 10000
    t_max = 0.5*np.pi
    x0 = np.array([1., 1., 1.])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    reg, j = 1, 0 # f vs. c indexing
    (X, Xdot, R) = ictg.sample(rng, N, t_max, reg, x0, x0dot)
    # m1 = X[R == 1,:].mean(axis=0) # mean is not the same
    # m2 = X[R == 2,:].mean(axis=0)
    s1 = X[R == 1,:].std(axis=0)
    s2 = X[R == 2,:].std(axis=0)
    error_s = s1 - s2
    print(s1, s2, error_s)
    assert np.all(np.abs(error_s) < 3/np.sqrt(N))

