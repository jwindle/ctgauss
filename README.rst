.. default-role:: math

Introduction
============

Our motivation comes from a question that arose when modeling dicot
(e.g. maize) root growth in a piecewise linear fashion and inferring
the path of a root given a point along its trajectory.

In its simplest form, the question is the following.  Suppose that a
priori `X \sim N(\mu, I_n / \phi)`.  Subsequently, we learn that
`\ell(X) = 0` where `\ell` is a piecewise affine,
continuous function.  How does one sample `(X | \ell(X) = 0)`?
We can, of course extend that to when the covariance is any positive
definite symmetric matrix `\Sigma`.

This software can sample from these sorts of distributions.  It uses
Hamilton Monte Carlo (HMC).

Please note though, that the excellent HMC software
[Stan](https://mc-stan.org/) can often sample from a nearly identical
distribution - so long as `\ell(x)` is not too complicated - by
insisting that `|\ell(x)| < \varepsilon`.  In other words, you
should check out Stan to see if you use-case fits there before
diving into this software.


Parameterization
================

There are two classes of main interest depending on if the covariance
structure is isotropic or anisotropic.  Those are
:code:`IsotropicCTGauss` and :code:`AnisotropicCTGauss`, respectively.
Both of these using the same parameters to define the function
`\ell(x)`.


Defining `\ell(x)`
------------------------

One defines the regions of `\ell(x)` using the following:

- (m, n) array `F`
- (m, 1) or (m,) array `g`
- (J, m) array `L`

The `i` th row of `F` and `i` th entry of `g` define a hyperplane via `f_i'
x + g = 0, i = 1, \ldots, m.`

The jth row of `L` defines how to construct the jth region, for
`j = 1, \ldots, J`.  The rules are

- The `i` th hyperplane is used (is active) in constructing the jth
  region if `L_{ji} \neq 0`.
- The jth region is defined by :math::
    \textmd{sign}(L_{ji}) (f_i' x + g) \geq 0
  for active `i`.
- If a particle in region `j` encounters the `i` th hyperplane, the
  magnitude of `L_{ji}` determines to which region it transitions.  In
  other words, for active `i`, the particle is reflected if `|L_{ji}|
  = j`, otherwise it passes through the hyperplane innto region
  `|L_{ji}|`.

  



Implementation Details
======================








