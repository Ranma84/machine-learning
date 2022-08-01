
import numpy as np
from numpy import linalg

from scipy.sparse import dok_matrix, csr_matrix, issparse
from scipy.spatial.distance import cosine, cityblock, minkowski


from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.metrics.pairwise import PAIRED_DISTANCES
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import check_paired_arrays
from sklearn.metrics.pairwise import _parallel_pairwise
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics.pairwise import paired_euclidean_distances
from sklearn.metrics.pairwise import paired_manhattan_distances
from sklearn.preprocessing import normalize

def test_pairwise_distances_argmin_min():
    # Check pairwise minimum distances computation for any metric
    X = [[0], [1]]
    Y = [[-1], [2]]

    X
    
    Xsp = dok_matrix(X)
    Ysp = csr_matrix(Y, dtype=np.float32)

    # euclidean metric
    D, E = pairwise_distances_argmin_min(X, Y, metric="euclidean")
    D2 = pairwise_distances_argmin(X, Y, metric="euclidean")
    assert_array_almost_equal(D, [0, 1])
    assert_array_almost_equal(D2, [0, 1])
    assert_array_almost_equal(D, [0, 1])
    assert_array_almost_equal(E, [1., 1.])

    # sparse matrix case
    Dsp, Esp = pairwise_distances_argmin_min(Xsp, Ysp, metric="euclidean")
    assert_array_equal(Dsp, D)
    assert_array_equal(Esp, E)
    # We don't want np.matrix here
    assert_equal(type(Dsp), np.ndarray)
    assert_equal(type(Esp), np.ndarray)

    # Non-euclidean sklearn metric
    D, E = pairwise_distances_argmin_min(X, Y, metric="manhattan")
    D2 = pairwise_distances_argmin(X, Y, metric="manhattan")
    assert_array_almost_equal(D, [0, 1])
    assert_array_almost_equal(D2, [0, 1])
    assert_array_almost_equal(E, [1., 1.])
    D, E = pairwise_distances_argmin_min(Xsp, Ysp, metric="manhattan")
    D2 = pairwise_distances_argmin(Xsp, Ysp, metric="manhattan")
    assert_array_almost_equal(D, [0, 1])
    assert_array_almost_equal(E, [1., 1.])

    # Non-euclidean Scipy distance (callable)
    D, E = pairwise_distances_argmin_min(X, Y, metric=minkowski,
                                         metric_kwargs={"p": 2})
    assert_array_almost_equal(D, [0, 1])
    assert_array_almost_equal(E, [1., 1.])

    # Non-euclidean Scipy distance (string)
    D, E = pairwise_distances_argmin_min(X, Y, metric="minkowski",
                                         metric_kwargs={"p": 2})
    assert_array_almost_equal(D, [0, 1])
    assert_array_almost_equal(E, [1., 1.])

    # Compare with naive implementation
    rng = np.random.RandomState(0)
    X = rng.randn(97, 149)
    Y = rng.randn(111, 149)

    dist = pairwise_distances(X, Y, metric="manhattan")
    dist_orig_ind = dist.argmin(axis=0)
    dist_orig_val = dist[dist_orig_ind, range(len(dist_orig_ind))]

    dist_chunked_ind, dist_chunked_val = pairwise_distances_argmin_min(
        X, Y, axis=0, metric="manhattan", batch_size=50)
    np.testing.assert_almost_equal(dist_orig_ind, dist_chunked_ind, decimal=7)
    np.testing.assert_almost_equal(dist_orig_val, dist_chunked_val, decimal=7)

    print("Distances")