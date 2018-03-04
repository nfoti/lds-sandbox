from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.lib.stride_tricks import as_strided


def _ensure_ndim(X, T, ndim):
    X = np.require(X,dtype=np.float64, requirements='C')
    assert ndim-1 <= X.ndim <= ndim
    if X.ndim == ndim:
        assert X.shape[0] == T
        return X
    else:
        return as_strided(X, shape=(T,)+X.shape, strides=(0,)+X.strides)


def rand_stable(d, s=0.9):
    A = np.random.randn(d, d)
    A *= s / np.max(np.abs(np.linalg.eigvals(A)))
    return A


def lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials):
    # write version that broadcasts over trials at some point

    d = A.shape[1]
    D = C.shape[0]

    x = np.empty((ntrials, T, d))
    y = np.empty((ntrials, T, D))

    L_R = np.linalg.cholesky(R)
    L_Q = np.linalg.cholesky(Q)

    L_Q = _ensure_ndim(L_Q, T, 3)

    for n in range(ntrials):
        x[n,0] = np.random.multivariate_normal(mu0, cov=Q0)
        y[n,0] = np.dot(C, x[n,0]) + np.dot(L_R, np.random.randn(D))

        for t in range(1, T):
            x[n,t] = np.dot(A[t-1], x[n,t-1]) + np.dot(L_Q[t-1], np.random.randn(d))
            y[n,t] = np.dot(C, x[n,t]) + np.dot(L_R, np.random.randn(D))

    return x, y
