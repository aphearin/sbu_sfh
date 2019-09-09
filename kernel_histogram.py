"""
"""
import numpy as np
from scipy.stats import norm


def kernel_histogram(data, bins, scale):
    """Calculate a Gaussian kernel histogram.

    Drop a gaussian kernel on top of each data point.
    For every point, calculate the probability that the point lies in each bin,
    by evaluating the CDF of the Gaussian associated with each point.
    Sum the results across all bins and return the result.

    Converges to an ordinary histogram in the limit of large data where scale << binsize.

    Parameters
    ----------
    data : ndarray of shape (ndata, )

    bins : ndarray of shape (nbins, )

    scale : float or ndarray of shape (ndata, )

    Returns
    -------
    khist : ndarray of shape (nbins-1, )
    """
    data = np.atleast_1d(data)
    bins = np.atleast_1d(bins)
    nbins, ndata = bins.size, data.size

    scale = np.zeros(ndata) + scale

    logsm_bin_matrix = np.repeat(bins, ndata).reshape((nbins, ndata)).astype('f4')
    data_matrix = np.tile(data, nbins).reshape((nbins, ndata)).astype('f4')
    smoothing_kernel_matrix = np.tile(scale, nbins).reshape((nbins, ndata)).astype('f4')

    cdf_matrix = norm.cdf(logsm_bin_matrix, loc=data_matrix, scale=smoothing_kernel_matrix)

    prob_bin_member = np.diff(cdf_matrix, axis=0)  #  Shape (nbins-1, ndata)

    total_num_bin_members = np.sum(prob_bin_member, axis=1)  #  Shape (nbins-1, )

    return total_num_bin_members
