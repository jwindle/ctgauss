# Copyright 2023 Jesse Windle

# Use of this source code is governed by an MIT-style license that can
# be found in the LICENSE file.


import numpy as np
import scipy as sp

from ctgauss import utils


def continuity_error(f, g, A1, A2, y1, y2):
    """Continuity error

    A'x - y error and projection error

    """
    B1 = np.hstack((A1, f))
    b1 = np.vstack((y1, g)) # A'x + y
    (n, d) = B1.shape
    Q, R = np.linalg.qr(B1, mode='complete')
    Q1 = Q[:,0:d]
    Q0 = Q[:,d: ]
    R1 = R[0:d,:]
    z1 = sp.linalg.solve_triangular(R1, b1, trans=1, lower=False) # A'x - y
    A2p = np.transpose(A2)
    e1 = np.matmul(A2p, np.matmul(Q1, z1)) - y2 # d x 1 # A'x - y
    e2 = np.matmul(A2p, Q0)                     # d x (n-d)
    ep1 = np.linalg.norm(e1) / np.sqrt(d)
    ep2 = np.linalg.norm(e2) / np.sqrt((n-d)*d)
    return (ep1, ep2)


def subspace_error(xp, x, xdot, Q, S, A, y):
    error = np.zeros((3,))
    if not ((A is None) or (len(A) == 0)):
        xpab = np.vstack([xp, x, xdot]).T
        x_all = np.hstack([xpab, Q, S])
        diffs = dict(
            xp = np.matmul(A.T, xp) - y, # A'x - y
            x = np.matmul(A.T, x),
            xdot = np.matmul(A.T, xdot),
            Q = np.matmul(A.T, Q),
            S = np.matmul(A.T, S)
        )
        errors = {k: np.linalg.norm(v) for k, v in diffs.items()}
        # diff = np.matmul(A.T, x_all)
        # diff[:,0] = diff[:,0] - y
        # error = np.linalg.norm(diff, axis=0)
    return errors


def region_error(x, F, g, A, y):
    e1 = -1. * (np.matmul(F, x) + g)
    e1 = np.where(e1 < 0., 0.0, e1)
    ep1 = np.sum(np.abs(e1))
    ep2 = 0.0
    if not ((A is None) or (len(A) == 0)):
        e2 = np.matmul(A.T, x) - y # A'x - y
        ep2 = np.sum(np.abs(e2))
    return (ep1, ep2)


def normal_velocity(x, f):
    u = f / np.linalg.norm(f)
    return np.dot(x, u)


def ode_coef(rng, xp, S, x0, x0dot):
    b = x0 - xp
    if x0dot is None:
        d = S.shape[1]
        ep = rng.standard_normal(d)
        a = np.matmul(S, ep)
    else:
        a = x0dot
    return (a, b)


def evolve_to_boundary(t_max, xp, a, b, F, g, l, j):
    Fa = np.matmul(F, a)
    Fb = np.matmul(F, b)
    phi = np.arctan2(-Fa, Fb)
    h = np.matmul(F, xp) + g
    u = np.sqrt(Fa**2 + Fb**2)
    # Several steps to make sure we have the right tau.
    tau = np.full(Fa.shape, fill_value=2*t_max+2*np.pi)
    np.arccos(-h/u, out=tau, where=(u >= np.abs(h)) & (u > 0))
    tau = tau - phi
    tau = np.where(tau < 0, tau + np.pi, tau)
    tau = np.where(tau > np.pi, tau - np.pi, tau)
    dK = Fa * np.cos(tau) - Fb * np.sin(tau)
    tau = np.where(dK > 0, tau + np.pi, tau)
    tau_star = t_max
    j_star = j
    f = a
    if np.any(tau <= t_max):
        # If you know you can't hit a corner:
        # k_star = np.argmin(tau)
        # j_star = np.abs(l[k_star]) - 1 # Since C-indexing
        # tau_star = tau[k_star]
        # f = F[k_star,:]
        # If there is a tie, go through the boundary that corresponds to the largest normal velocity
        tau_star, dK_star, k_star = utils.minamin2d(tau, dK)
        j_star = np.abs(l[k_star]) - 1 # j_star is in c-indexing!
        f = F[k_star]
        # Could check that dK_star is negative here
    x = xp + a * np.sin(tau_star) + b * np.cos(tau_star)
    xdot = a * np.cos(tau_star) - b * np.sin(tau_star)
    return(tau_star, j_star, f, x, xdot)


def wall_dynamics(xdot, u1):
    v1 = np.dot(u1, xdot)
    xdot_new = xdot - 2 * v1 * u1
    return xdot_new


def boundary_dynamics(xdot, u1, u2, V1, V2, j1, j2):
    v1 = np.dot(u1, xdot)
    Ev1 = 0.5 * v1**2
    DeltaV = V2 - V1
    xdot_new = xdot - v1 * u1
    if Ev1 < DeltaV: # strictly less than is critical for fake/hard boundary trick
        # Reflect
        j_new = j1
        xdot_new = xdot_new - v1 * u1
    else:
        # Refract
        j_new = j2
        xdot_new = xdot_new + np.sqrt(2.*(Ev1 - DeltaV)) * u2
    return (xdot_new, j_new)


