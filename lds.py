from __future__ import division
from __future__ import print_function

import numpy as np
from autograd_linalg import solve_triangular
from einsum2 import einsum2, batched_dot


def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))


def rand_stable(d):
    A = np.random.randn(d, d)
    A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
    return A


def lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials):
    # write version that broadcasts over trials at some point

    d = A.shape[1]
    D = C.shape[0]

    x = np.empty((ntrials, T, d))
    y = np.empty((ntrials, T, D))

    L_R = np.linalg.cholesky(R)
    L_Q = np.linalg.cholesky(Q)

    for n in range(ntrials):
        x[n,0] = np.random.multivariate_normal(mu0, cov=Q0)
        y[n,0] = np.dot(C, x[n,0]) + np.dot(L_R, np.random.randn(D))

        for t in range(1, T):
            x[n,t] = np.dot(A[t-1], x[n,t-1]) + np.dot(L_Q[t-1], np.random.randn(d))
            y[n,t] = np.dot(C, x[n,t]) + np.dot(L_R, np.random.randn(D))

    return x, y


def kalman_filter_loop(Y, A, C, Q, R, mu0, Q0):
    """ Kalman filter that broadcasts over the first dimension.
        
        Note: This function doesn't handle control inputs (yet).
        
        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D, D)
          Time-varying dynamics matrices
        
        C : ndarray, shape=(p, D)
          Observation matrix

        Q : ndarray, shape=(T, D, D)
          Covariance of latent states
        
        R : ndarray, shape=(T, D, D)
          Covariance of observations

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable
    """

    N = Y.shape[0]
    T, D, _ = A.shape
    p = C.shape[0]

    mu_predict = np.stack([mu0 for _ in range(N)], axis=0)
    sigma_predict = np.stack([Q0 for _ in range(N)], axis=0)

    mus_filt = np.zeros((N, T, D))
    sigmas_filt = np.zeros((N, T, D, D))

    ll = 0.

    for n in range(N):
        for t in range(T):

            # condition
            tmp1 = np.dot(C, sigma_predict[n])
            sigma_pred = np.dot(tmp1, C.T) + R
            L = np.linalg.cholesky(sigma_pred)
            v = solve_triangular(L, Y[n,t,:] - np.dot(C, mu_predict[n]))

            # log-likelihood over all trials
            ll += -0.5*np.dot(v,v) - np.sum(np.log(np.diag(L))) \
                  - 0.5*np.log(2.*np.pi)

            mus_filt[n,t,:] = mu_predict[n] + np.dot(tmp1.T, solve_triangular(L, v, 'T'))

            tmp2 = solve_triangular(L, tmp1)
            sigmas_filt[n,t,:,:] = sigma_predict[n] - np.dot(tmp2.T, tmp2)

            # prediction
            mu_predict[n] = np.dot(A[t], mus_filt[n,t,:])
            sigma_predict[n] = dot3(A[t], sigmas_filt[n,t,:,:], A[t].T) + Q[t]

    return ll, mus_filt, sigmas_filt


def kalman_filter(Y, A, C, Q, R, mu0, Q0):
    """ Kalman filter that broadcasts over the first dimension.
        
        Note: This function doesn't handle control inputs (yet).
        
        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D, D)
          Time-varying dynamics matrices
        
        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(T, D, D)
          Covariance of latent states
        
        R : ndarray, shape=(T, D, D)
          Covariance of observations
    """

    N = Y.shape[0]
    T, D, _ = A.shape
    d = C.shape[0]

    mu_predict = np.stack([mu0 for _ in range(N)], axis=0)
    sigma_predict = np.stack([Q0 for _ in range(N)], axis=0)

    mus_filt = np.zeros((N, T, D))
    sigmas_filt = np.zeros((N, T, D, D))

    ll = 0.

    for t in range(T):

        # condition
        # dot3(C, sigma_predict, C.T) + R
        #tmp1 = np.einsum('ik,nkj->nij', C, sigma_predict)
        tmp1 = einsum2('ik,nkj->nij', C, sigma_predict)
        sigma_pred = np.dot(tmp1, C.T) + R

        L = np.linalg.cholesky(sigma_pred)
        # res[n] = Y[n,t,:] = np.dot(C, mu_predict[n])
        # the transpose works b/c of how dot broadcasts
        res = Y[...,t,:] - np.dot(mu_predict, C.T)
        v = solve_triangular(L, res)
        
        # log-likelihood over all trials
        ll += (-0.5*np.sum(v*v)
               - np.sum(np.log(np.diagonal(L, axis1=1, axis2=2))) 
               - N/2.*np.log(2.*np.pi))

        #mus_filt[...,t,:] = mu_predict + np.einsum('nki,nk->ni', tmp1, solve_triangular(L, v, 'T'))
        mus_filt[...,t,:] = mu_predict + einsum2('nki,nk->ni', tmp1, solve_triangular(L, v, 'T'))

        tmp2 = solve_triangular(L, tmp1)
        #sigmas_filt[...,t,:,:] = sigma_predict - np.einsum('nki,nkj->nij', tmp2, tmp2)
        sigmas_filt[...,t,:,:] = sigma_predict - einsum2('nki,nkj->nij', tmp2, tmp2)

        # prediction
        #mu_predict = np.dot(A[t], mus_filt[t])
        #mu_predict = np.einsum('ik,nk->ni', A[t], mus_filt[...,t,:])
        mu_predict = einsum2('ik,nk->ni', A[t], mus_filt[...,t,:])

        #sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]
        #sigma_predict = np.einsum('ik,nkl,jl->nij', A[t], sigmas_filt[...,t,:,:], A[t]) + Q[t]
        sigma_predict = einsum2('ik,nkl->nil', A[t], sigmas_filt[...,t,:,:])
        sigma_predict = einsum2('nil,jl->nij', sigma_predict, A[t]) + Q[t]

    return ll, mus_filt, sigmas_filt


if __name__ == "__main__":

    #np.random.seed(8675309)
    np.random.seed(42)

    T = 165
    ntrials = 150
    #theta = 1.2
    #A = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    # the same for convenience. constructing reasonable C from small to large
    # dimensions is tricky
    d = 5
    D = 6

    A = rand_stable(d)
    A = np.stack([A for _ in range(T)], axis=0)

    #C = np.eye(D)
    C, _ = np.linalg.qr(np.random.randn(D, d))

    Q0 = 0.2*np.eye(d)
    Q = np.stack([0.1*np.eye(d) for _ in range(T)], axis=0)

    R = 0.05*np.eye(D)

    mu0 = np.zeros(d)

    x, Y = lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials)

    ll_loop, mus_filt_loop, sigmas_filt_loop = kalman_filter_loop(Y, A, C, Q, R, mu0, Q0)
    ll, mus_filt, sigmas_filt = kalman_filter(Y, A, C, Q, R, mu0, Q0)

    assert np.allclose(ll, ll_loop)
    assert np.allclose(mus_filt, mus_filt_loop)
    assert np.allclose(sigmas_filt, sigmas_filt_loop)
