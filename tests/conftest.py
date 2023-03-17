# Copyright 2023 Jesse Windle

# Use of this source code is governed by an MIT-style license that can
# be found in the LICENSE file.


import pytest
import numpy as np

from ctgauss import IsotropicCTGauss, AnisotropicCTGauss


def fullspace_3d_param():
    F = np.array([
        [1, 0, 0],
    ])
    g = np.array([0])
    L = np.array([
        [2],
        [-1]
    ])
    A1 = np.array([-1, 1, 0]).reshape(3,1)
    A2 = np.array([ 1, 1, 0]).reshape(3,1)
    A = [A1, A2]
    y1 = np.array([0])
    y2 = np.array([0])
    y = [y1, y2]
    param = (A, y, F, g, L)
    return param


def halfspace_3d_param():
    F = np.array([
        [1, 0, 0],
        [0, 0, 1]
    ])
    g = np.array([0, 0])
    L = np.array([
        [2, 1],
        [-1, 2]
    ])
    A1 = np.array([-1, 1, 0]).reshape(3,1)
    A2 = np.array([ 1, 1, 0]).reshape(3,1)
    A = [A1, A2]
    y1 = np.array([0])
    y2 = np.array([0])
    y = [y1, y2]
    param = (A, y, F, g, L)
    return param


def halfspace_3d_param_as_array():
    F = np.array([
        [1, 0, 0],
        [0, 0, 1]
    ])
    g = np.array([0, 0])
    L = np.array([
        [2, 1],
        [-1, 2]
    ])
    A1 = np.array([-1, 1, 0]).reshape(3,1)
    A2 = np.array([ 1, 1, 0]).reshape(3,1)
    A = np.stack([A1, A2])
    y1 = np.array([0])
    y2 = np.array([0])
    y = np.stack([y1, y2])
    param = (A, y, F, g, L)
    return param


fullspace_param = fullspace_3d_param()
halfspace_param = halfspace_3d_param()
halfspace_param_as_array = halfspace_3d_param_as_array()


# @pytest.fixture(scope='module')
# def isotropic_full_example(fullspace_3d_param):
#     mu = np.array([0., 0., 0.])
#     phi = 1.0
#     (A, y, F, g, L) = fullspace_3d_param
#     ictg = IsotropicCTGauss(phi, mu, A, y, F, g, L)
#     return ictg


# @pytest.fixture(scope='module')
# def isotropic_half_example(halfspace_3d_param):
#     mu = np.array([0., 0., 0.])
#     phi = 1.0
#     (A, y, F, g, L) = halfspace_3d_param
#     ictg = IsotropicCTGauss(phi, mu, A, y, F, g, L)
#     return ictg


@pytest.fixture(scope='module')
def rng():
    rng = np.random.default_rng()
    return rng


@pytest.fixture(
    scope='module',
    params=[(fullspace_param, 1), (fullspace_param, 2)],
    ids=["iso", "aniso"]
)
def isotropic_fullspace(request):
    (A, y, F, g, L) = request.param[0]
    mu = np.array([0., 0., 0.])
    phi = 0.5
    if request.param[1] == 1:
        ictg = IsotropicCTGauss(phi, mu, A, y, F, g, L)
    else:
        Prec = phi * np.eye(3)
        ictg = AnisotropicCTGauss(Prec, mu, A, y, F, g, L, mean=True)
    return ictg, (phi, mu), request.param[0]


@pytest.fixture(
    scope='module',
    params=[(halfspace_param, 1), (halfspace_param, 2), (halfspace_param_as_array, 1)],
    ids=["iso-list", "aniso-list", "iso-array"]
)
def isotropic_halfspace(request):
    (A, y, F, g, L) = request.param[0]
    mu = np.array([0., 0., 0.])
    phi = 1.0
    if request.param[1] == 1:
        ictg = IsotropicCTGauss(phi, mu, A, y, F, g, L)
    else:
        Prec = phi * np.eye(3)
        ictg = AnisotropicCTGauss(Prec, mu, A, y, F, g, L, mean=True)
    return ictg, (phi, mu), request.param[0]


@pytest.fixture(
    scope='module',
    params=[(halfspace_param, 1), (halfspace_param, 2)],
    ids=["iso", "aniso"]
)
def isotropic_halfspace_sym_reg(request):
    (A, y, F, g, L) = request.param[0]
    mu = np.array([0., 1., 0.])
    phi = 0.5
    if request.param[1] == 1:
        ictg = IsotropicCTGauss(phi, mu, A, y, F, g, L)
    else:
        Prec = phi * np.eye(3)
        ictg = AnisotropicCTGauss(Prec, mu, A, y, F, g, L, mean=True)
    return ictg, (phi, mu), request.param[0]


