"""
MDBSCAN: Modified Density-Based Spatial Clustering of Applications with Noise
"""

# Based on the DBSCAN implementation in sci-kit learn whose authors are
#         Robert Layton <robertlayton@gmail.com>
#         Joel Nothman <joel.nothman@gmail.com>
#         Lars Buitinck
#         
# The DBSCAN implementation was modified into the MDBSCAN algorithm by Viktor Rucký
#
# MDBSCAN is introduced in the paper
# Mangling Rules Generation With Density-Based Clustering for Password Guessing (DOI: 10.1109/TDSC.2022.3217002)
# by Shunbin Li, Zhiyu Wang, Ruyun Zhang, Chunming Wu, and Hanguang Luo
#
# License: BSD 3 clause

import warnings
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import _VALID_METRICS
from ..neighbors import NearestNeighbors
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _check_sample_weight
from ._mdbscan_inner import mdbscan_inner


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=False,
)
def mdbscan(
    X,
    y=None,
    eps1=0.5,
    eps2=0.25,
    *,
    min_samples=5,
    metric1="minkowski",
    metric1_params=None,
    metric2="minkowski",
    metric2_params=None,
    algorithm="auto",
    leaf_size=30,
    p=2,
    sample_weight=None,
    n_jobs=None,
):
    """Perform MDBSCAN clustering from vector array or distance matrix.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    X : {array-like, sparse (CSR) matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    eps1 : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : str or callable, default='minkowski'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.
        X may be a :term:`sparse graph <sparse graph>`,
        in which case only "nonzero" elements may be considered neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=2
        The power of the Minkowski metric to be used to calculate distance
        between points.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight of each sample, such that a sample with a weight of at least
        ``min_samples`` is by itself a core sample; a sample with negative
        weight may inhibit its eps-neighbor from being core.
        Note that weights are absolute, and default to 1.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. ``None`` means
        1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors. See :term:`Glossary <n_jobs>` for more details.
        If precomputed distance are used, parallel execution is not available
        and thus n_jobs will have no effect.

    Returns
    -------
    core_samples : ndarray of shape (n_core_samples,)
        Indices of core samples.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.  Noisy samples are given the label -1.

    See Also
    --------
    DBSCAN : An estimator interface for this clustering algorithm.
    OPTICS : A similar estimator interface clustering at multiple values of
        eps. Our implementation is optimized for memory usage.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_dbscan.py
    <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower
    memory usage.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
    <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
    :doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
    <10.1145/3068335>`
    ACM Transactions on Database Systems (TODS), 42(3), 19.

    Examples
    --------
    >>> from sklearn.cluster import dbscan
    >>> X = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    >>> core_samples, labels = dbscan(X, eps=3, min_samples=2)
    >>> core_samples
    array([0, 1, 2, 3, 4])
    >>> labels
    array([ 0,  0,  0,  1,  1, -1])
    """

    est = MDBSCAN(
        eps1=eps1,
        eps2=eps2,
        min_samples=min_samples,
        metric1=metric1,
        metric1_params=metric1_params,
        metric2=metric2,
        metric2_params=metric2_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=n_jobs,
    )
    est.fit(X, y, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_


class MDBSCAN(ClusterMixin, BaseEstimator):
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    This implementation has a worst case memory complexity of :math:`O({n}^2)`,
    which can occur when the `eps` param is large and `min_samples` is low,
    while the original DBSCAN only uses linear memory.
    For further details, see the Notes below.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself. If
        `min_samples` is set to a higher value, DBSCAN will find denser clusters,
        whereas if it is set to a lower value, the found clusters will be more
        sparse.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.

        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=None
        The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.

    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    OPTICS : A similar clustering at multiple values of eps. Our implementation
        is optimized for memory usage.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_dbscan.py
    <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower memory
    usage.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
    <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
    :doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
    <10.1145/3068335>`
    ACM Transactions on Database Systems (TODS), 42(3), 19.

    Examples
    --------
    >>> from sklearn.cluster import DBSCAN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])
    >>> clustering
    DBSCAN(eps=3, min_samples=2)
    """

    #TODO: We should check that if y is used, that the sizes of X and y are in agreement
    _parameter_constraints: dict = {
        "eps1": [Interval(Real, 0.0, None, closed="neither")],
        "eps2": [Interval(Real, 0.0, None, closed="neither")],
        "min_samples": [Interval(Integral, 1, None, closed="left")],
        "metric1": [
            StrOptions(set(_VALID_METRICS) | {"precomputed"}),
            callable,
        ],
        "metric2": [
            StrOptions(set(_VALID_METRICS) | {"precomputed"}),
            callable,
        ],
        "metric1_params": [dict, None],
        "metric2_params": [dict, None],
        "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0.0, None, closed="left"), None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        eps1=0.5,
        eps2=0.25,
        *,
        min_samples=5,
        metric1="euclidean",
        metric1_params=None,
        metric2="minkowski",
        metric2_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric1 = metric1
        self.metric1_params = metric1_params
        self.metric2 = metric2
        self.metric2_params = metric2_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    @_fit_context(
        # DBSCAN.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, sample_weight=None):
        """Perform MDBSCAN clustering from features, or distance matrices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : 
            Other distance matrix

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        X = self._validate_data(X, accept_sparse="csr")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric1 == "precomputed" and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            X = X.copy()  # copy to avoid in-place modification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())

        if self.metric2 == "precomputed" and sparse.issparse(y):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            y = y.copy()  # copy to avoid in-place modification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                y.setdiag(y.diagonal())

        neighbors_model1 = NearestNeighbors(
            radius=self.eps1,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric1,
            metric_params=self.metric1_params,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        neighbors_model1.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods1 = neighbors_model1.radius_neighbors(X, return_distance=False)

        #TODO: make it work also with non precomputed metrics
        neighbors_model2 = NearestNeighbors(
            radius=self.eps2,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric2,
            metric_params=self.metric2_params,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        neighbors_model2.fit(y)
        # This has worst case O(n^2) memory complexity
        neighborhoods2 = neighbors_model2.radius_neighbors(y, return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods1])
        else:
            n_neighbors = np.array(
                [np.sum(sample_weight[neighbors]) for neighbors in neighborhoods1]
            )

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
        mdbscan_inner(core_samples, neighborhoods1, neighborhoods2, labels)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute clusters from a data or distance matrix and predict labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_

    def _more_tags(self):
        return {"pairwise": self.metric1 == "precomputed"}
