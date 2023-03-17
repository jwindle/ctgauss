# Copyright 2023 Jesse Windle

# Use of this source code is governed by an MIT-style license that can
# be found in the LICENSE file.


import numpy as np


def test_dimension(isotropic_fullspace):
    ictg, phimu, param = isotropic_fullspace
    assert ictg.dim == 3
    assert ictg.dim_reduction == 1


def test_continuity(isotropic_fullspace):
    ictg, phimu, param = isotropic_fullspace
    error_array = ictg.continuity_error()
    assert np.linalg.norm(error_array) < 1e-12


def test_evolve1(isotropic_fullspace, rng):
    ictg, phimu, param = isotropic_fullspace
    t_max = 100
    j = 0 # Region 0 for the purposes of evolve, but we call it region 1
    x0 = np.array([1., 1., 0.])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    (tau, j, x, xdot) = ictg.evolve1(rng, t_max, j, x0, x0dot)
    udot = xdot / np.linalg.norm(xdot)
    assert np.all(x == 0.)
    assert np.dot(x0dot, xdot) < 1e-15
    assert j == 1


def test_evolve2(isotropic_fullspace, rng):
    ictg, phimu, param = isotropic_fullspace
    t_max = 100
    j = 0 # Region 0 for the purposes of evolve, but we call it region 1
    x0 = np.array([1., 1., 0.])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    (tau, j, x, xdot) = ictg.evolve2(rng, t_max, j, x0, x0dot)
    udot = xdot / np.linalg.norm(xdot)
    assert np.all(x == 0.)
    assert np.dot(x0dot, xdot) < 1e-15
    assert j == 1


def test_sample_runs(isotropic_fullspace, rng):
    ictg, phimu, param = isotropic_fullspace
    N = 1000
    t_max = 0.5*np.pi
    x0 = np.array([1., 1., 0.0])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    reg, j = 1, 0
    (X, Xdot, R) = ictg.sample(rng, N, t_max, reg, x0, x0dot)
    mean_x_3 = np.mean(X[:,2])
    var_x_12 = (X[:,0:2]**2).sum(axis=1).mean(axis=0)
    var_x_3 = np.mean(X[:,2]**2, axis=0)
    error_var = np.array([var_x_12, var_x_3]) - 1./phimu[0]
    print((var_x_12, var_x_3), error_var)
    assert np.abs(mean_x_3) < 10/np.sqrt(N)
    assert np.all(np.abs(error_var) < 10/np.sqrt(N))


def test_sample_with_boundaries_runs(isotropic_fullspace, rng):
    ictg, phimu, param = isotropic_fullspace
    N = 1000
    t_max = 0.5*np.pi
    x0 = np.array([1., 1., 0.0])
    x0dot = np.array([-1., -1., 0.]) / np.sqrt(2)
    reg, j = 1, 0
    (X, Xdot, R, I) = ictg.sample_with_boundaries(rng, N, t_max, reg, x0, x0dot)
    N_interior = np.sum(I)
    N_boundary = N - N_interior
    mean_x_3 = np.mean(X[I,2])
    assert np.abs(mean_x_3) < 10/np.sqrt(N_interior)


