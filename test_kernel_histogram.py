"""
"""
import numpy as np
from kernel_histogram import numpy_kernel_histogram, numba_kernel_histogram


def test1():
    nbins, ndata = 10, 500
    bins = np.linspace(0, 1, nbins)
    data = np.linspace(-1, 2, ndata)
    scale = 0.1

    numpy_khist = numpy_kernel_histogram(data, bins, scale)
    numba_khist = np.empty_like(numpy_khist)
    numba_kernel_histogram(data, bins, scale, numba_khist)
    assert np.allclose(numpy_khist, numba_khist, rtol=1e-3)
