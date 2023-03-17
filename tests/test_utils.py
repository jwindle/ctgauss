# Copyright 2023 Jesse Windle                                                                                                                                                                                                           

# Use of this source code is governed by an MIT-style                                                                                                                                                                                   
# license that can be found in the LICENSE file.


import numpy as np

from ctgauss import utils


def test_minamin2d():
    x = np.array([1, 2, 3, 4, 3, 2, 1])
    y = np.array([1, 2, 3, 4, 5, 6, 7])
    (xstar, ystar, istar) = utils.minamin2d(x, y)
    assert istar == 0
    assert xstar == 1
    assert ystar == 1
    (xstar, ystar, istar) = utils.minamin2d(x, -y)
    assert istar == 6
    assert xstar == 1
    assert ystar == -7
