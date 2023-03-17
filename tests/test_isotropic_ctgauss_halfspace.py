# Copyright 2023 Jesse Windle

# Use of this source code is governed by an MIT-style license that can
# be found in the LICENSE file.


import numpy as np
import pytest


def test_dimension(isotropic_halfspace):
    ictg, phimu, param = isotropic_halfspace
    assert ictg.dim == 3
    assert ictg.dim_reduction == 1


def test_continuity(isotropic_halfspace):
    ictg, phimu, param = isotropic_halfspace
    error_array = ictg.continuity_error()
    assert np.linalg.norm(error_array) < 1e-12


def test_evolve1(isotropic_halfspace, rng):
    ictg, phimu, param = isotropic_halfspace
    t_max = 100
    j = 0 # Region 0 for the purposes of evolve, but we call it region 1
    x0 = np.array([1., 1., 1.])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    (tau, j, x, xdot) = ictg.evolve1(rng, t_max, j, x0, x0dot)
    testdot = np.array([1., 1., 0.])
    assert x[0] == 0 and x[1] == 0
    assert np.dot(testdot, xdot) < 1e-15
    assert j == 1 # c-indexing


def test_evolve2(isotropic_halfspace, rng):
    ictg, phimu, param = isotropic_halfspace
    t_max = 100
    j = 0 # Region 0 for the purposes of evolve, but we call it region 1
    x0 = np.array([1., 1., 1.])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    (tau, j, x, xdot) = ictg.evolve2(rng, t_max, j, x0, x0dot)
    testdot = np.array([1., 1., 0.])
    assert x[0] == 0 and x[1] == 0
    assert np.dot(testdot, xdot) < 1e-15
    assert j == 1 # c-indexing


def test_corner(isotropic_halfspace, rng):
    ictg, phimu, param = isotropic_halfspace
    t_max = 100
    j0 = 0 # Region 0 for the purposes of evolve, but we call it region 1
    x0 = np.array([1., 1., 1.])
    x0dot = np.array([-1., -1., -1.]) / np.sqrt(3)
    (tau1, j1, x1, x1dot) = ictg.evolve1(rng, t_max, j0, x0, x0dot)
    (tau2, j2, x2, x2dot) = ictg.evolve1(rng, t_max, j1, x1, x1dot)
    # print(t_max, j0, x0, x0dot)
    # print(tau1, j1, x1, x1dot)
    # print(tau2, j2, x2, x2dot)
    assert j1 == 1 # c-indexing
    assert j2 == 1 # c-indexing
    assert np.abs(x1dot[0] - (-x1dot[1])) < 1e-12
    assert np.abs(x1dot[0] - x1dot[2]) < 1e-12
    assert np.abs(x1dot[0] - x2dot[0]) < 1e-12
    assert np.abs(x1dot[1] - x2dot[1]) < 1e-12
    assert np.abs(-x1dot[2] - x2dot[2]) < 1e-12

    
def test_sample_runs(isotropic_halfspace, rng):
    ictg, phimu, param = isotropic_halfspace
    N = 1000
    t_max = 0.5*np.pi
    x0 = np.array([1., 1., 1.])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    reg, j = 1, 0 # f vs. c indexing
    (X, Xdot, R) = ictg.sample(rng, N, t_max, reg, x0, x0dot)
    r_sign = 2 * (np.array(R) - 1) - 1 # R is in f-indexing
    r = r_sign * np.linalg.norm(X[:,0:2], axis=1)
    mean_r = np.mean(r)
    mean_x_3 = np.mean(X[:,2])
    assert np.abs(mean_r) < 10/np.sqrt(N)
    assert np.abs(mean_x_3 - 1) < 10/np.sqrt(N)


def test_sample_with_boundaries_runs(isotropic_halfspace, rng):
    ictg, phimu, param = isotropic_halfspace
    N = 1000
    t_max = 0.5*np.pi
    x0 = np.array([1., 1., 0.0])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    reg, j = 1, 0
    (X, Xdot, R, I) = ictg.sample_with_boundaries(rng, N, t_max, reg, x0, x0dot)
    N_interior = np.sum(I)
    N_boundary = N - N_interior
    print(R[0:10])
    r_sign = 2 * (np.array(R) - 1) - 1 # R is in f-indexing
    r = r_sign * np.linalg.norm(X[:,0:2], axis=1)
    mean_r = np.mean(r[I])
    mean_x_3 = np.mean(X[I,2])
    assert np.abs(mean_r) < 10/np.sqrt(N)
    assert np.abs(mean_x_3 - 1) < 10/np.sqrt(N_interior)


def test_break_subspace_constraint(isotropic_halfspace, rng):
    ictg, phimu, param = isotropic_halfspace
    t_max = 0.5 * np.pi
    j = 0 # Region 0 for the purposes of evolve, but we call it region 1
    x0 = np.array([1. + 1e-10, 1., 1.])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    with pytest.raises(Exception):
        (tau, j, x, xdot) = ictg.evolve1(rng, t_max, j, x0, x0dot, tol=1e-32)
    # assert False
