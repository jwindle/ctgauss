# Copyright 2023 Jesse Windle                                                                                     

# Use of this source code is governed by an MIT-style license that can
# be found in the LICENSE file.


import numpy as np
from abc import ABC, abstractmethod

from ctgauss import dynamics
from ctgauss import utils


class CTGaussBase(ABC):

    def __init__(self, A, y, F, g, L):
        """ y is b """
        self.J = L.shape[0]
        self.param_cache = [None] * self.J
        self.boundary_cache = [None] * self.J
        self.dim = F.shape[1]
        self.dim_reduction = utils.constraint_dimension(A, y, self.dim)
        self.A = A
        self.y = y
        self.F = F
        self.g = g.squeeze() if len(g.shape) > 1 else g
        self.L = L.astype(int) # Float will cause error in indexing.


    @abstractmethod
    def get_ode_params(self, j):
        pass


    @abstractmethod
    def get_potential(self, j, x):
        pass


    def get_boundaries(self, j):
        active = self.L[j,:] != 0
        L_j = self.L[j,active]
        n_active = L_j.shape[0]
        signs = np.sign(L_j)
        F_j = self.F[active,:] * signs.reshape(n_active, 1)
        g_j = self.g[active] * signs
        return (F_j, g_j, L_j)


    def cached_ode_params(self, j):
        if not self.param_cache[j]:
            self.param_cache[j] = self.get_ode_params(j)
        return self.param_cache[j]


    def cached_boundaries(self, j):
        # j is in c-indexing
        if not self.boundary_cache[j]:
            self.boundary_cache[j] = self.get_boundaries(j)
        return self.boundary_cache[j]
        

    def continuity_error(self):
        # This only makes sense when we have conditional contstraint Ax=y
        # Also: do we want to change to Ax + y = 0, currently Ax - y = 0
        # J = num of regions
        # m = num of constraints
        J, m = self.L.shape
        error_array = np.zeros((2, J, m))
        if self.dim_reduction == 0:
            return error_array
        for j in range(J):
            for i in range(m):
                if self.L[j,i] != 0:
                    j_star = np.abs(self.L[j,i]) - 1 # Since Fortran-indexing
                    f = self.F[i,:].reshape((self.F.shape[1], 1))
                    g = self.g[i]
                    A1 = self.A[j]
                    A2 = self.A[j_star]
                    y1 = self.y[j]
                    y2 = self.y[j_star]
                    if j == j_star: # reflection
                        continue
                    try:
                        error_ij = dynamics.continuity_error(f, g, A1, A2, y1, y2)
                    except Exception as err:
                        print(f"Problem computing cotinuity error from region {j+1} to region {j_star+1}")
                        print(f"f={f}")
                        print(f"g={g}")
                        print(f"A1={A1}")
                        print(f"A2={A2}")
                        print(f"y1={y1}")
                        print(f"y2={y2}")
                        raise(err)
                    error_array[0,j,i] = error_ij[0]
                    error_array[1,j,i] = error_ij[1]
        return error_array


    def assert_location_and_velocity(self, j, x, xdot, tol=1e-10):
        d = self.dim_reduction
        Fj, gj, Lj = self.cached_boundaries(j)
        re = dynamics.region_error(x, Fj, gj, self.A[j], self.y[j])
        if (re[0] > tol) or (re[1] > tol):
            xp, Q, S = self.cached_ode_params(j)
            se = dynamics.subspace_error(xp, x, xdot, Q[:,d:], S, self.A[j], self.y[j])
            error_str = (
                f"Particle outside region or constraint for region {j+1}.\n"
                f"Location = {x}\n"
                f"Velocity = {xdot}\n"
                f"Error(region, subspace) = {re}.\n"
                f"Subspace error decomp: {se}"
            )
            raise Exception(error_str)
    
    
    def evolve1(self, rng, t_max, j, x0, x0dot, tol=1e-10):
        # j is in c-indexing
        xp, Q, S = self.cached_ode_params(j)
        d = self.dim_reduction
        a, b = dynamics.ode_coef(rng, xp, S, x0, x0dot)
        Fj, gj, Lj = self.cached_boundaries(j)
        tau_star, j_star, f, x, xdot = dynamics.evolve_to_boundary(t_max, xp, a, b, Fj, gj, Lj, j)
        # TODO: do we want to make sure x is in the space?
        if tau_star < t_max:
            u1 = utils.resid(f, d, Q, normalize=True)
            if j == j_star:
                xdot = dynamics.wall_dynamics(xdot, u1)
                if np.dot(xdot, u1) < -tol:
                    raise Exception("Particle is not traveling back into region, but it supposed to be reflecting")
            else:
                xp, Q, S = self.cached_ode_params(j_star)
                u2 = utils.resid(-f, d, Q, normalize=True)
                V1 = self.get_potential(j, x)
                V2 = self.get_potential(j_star, x)
                xdot, j_star = dynamics.boundary_dynamics(xdot, u1, u2, V1, V2, j, j_star)
                if np.dot(xdot, u2) < -tol:
                    raise Exception("Particle is not traveling out of region, but it supposed to be refracting")
        # Lastly, make sure we are in the correct region.
        self.assert_location_and_velocity(j_star, x, xdot, tol)
        return (tau_star, j_star, x, xdot)
    
    
    def evolve2(self, rng, t_max, j, x0, x0dot):
        xp, Q, S = self.cached_ode_params(j)
        d = self.dim_reduction
        a, b = dynamics.ode_coef(rng, xp, S, x0, x0dot)
        Fj, gj, Lj = self.cached_boundaries(j)
        tau_star, j_star, f, x, xdot = dynamics.evolve_to_boundary(t_max, xp, a, b, Fj, gj, Lj, j)
        u1 = utils.resid(f, d, Q, normalize=True)
        xp, Q, S = self.cached_ode_params(j_star)
        pm = 1 if (tau_star < t_max) and (j == j_star) else -1
        u2 = pm * utils.resid(f, d, Q, normalize=True)
        V1 = self.get_potential(j, x)
        V2 = self.get_potential(j_star, x)
        xdot, jstar = dynamics.boundary_dynamics(xdot, u1, u2, V1, V2, j, j_star)
        return (tau_star, j_star, x, xdot)
        
    
    def sample(self, rng, N, t_max, reg, x0, x0dot):
        # move reg to default None and then find region?
        j = reg - 1 # reg is in f-indexing
        n = len(x0)
        X = np.zeros((N, n))
        Xdot = np.zeros((N, n))
        R = np.zeros((N,))
        j_prev = j
        x = x0
        xdot = x0dot
        t = t_max
        self.assert_location_and_velocity(j, x, xdot)
        for i in range(N):
            while t > 0: # Need to prevent infinite loop
                # j_prev = j
                x_prev = x
                xdot_prev = xdot
                tau, j, x, xdot = self.evolve1(rng, t, j, x, xdot)
                if np.all(x == x_prev) and np.all(xdot == xdot_prev):
                    #  print(tau, j, x, xdot)
                    raise Exception(f"Algorithmic fixed point in sample {i}")
                t = t - tau
            X[i,:] = x
            Xdot[i,:] = xdot
            R[i] = j + 1 # Return to f-indexing
            t = t_max
            xdot = None
        return (X, Xdot, R)


    def sample_with_boundaries(self, rng, N, t_max, reg, x0, x0dot):
        j = reg - 1
        n = len(x0)
        X = np.zeros((N, n))
        Xdot = np.zeros((N, n))
        R = np.zeros((N,))
        I = np.zeros((N,), dtype=bool)
        x = x0
        xdot = x0dot
        t = t_max
        self.assert_location_and_velocity(j, x, xdot)
        for i in range(N):
            tau, j, x, xdot = self.evolve1(rng, t, j, x, xdot)
            # self.assert_location_and_velocity(j, x, xdot, tol)
            t = t - tau
            X[i,:] = x
            Xdot[i,:] = xdot
            R[i] = j + 1
            I[i] = t == 0
            if False:
                print(t, x, xdot, R[i], I[i])
            if t == 0:
                t = t_max
                xdot = None
        return (X, Xdot, R, I)
