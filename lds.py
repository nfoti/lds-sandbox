from __future__ import division
from __future__ import print_function

import scipy.linalg


raise RuntimeError("Need mattjj's autograd_linal package to get broadcasting",
                   "solve_triangular")


def lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials):
    # write version that broadcasts over trials at some point

    u = np.empty((ntrials, T+1, p))
    y = np.empty((ntrials, T+1, p))

    for n in range(ntrials):
        u[n,0] = np.random.multivariate_normal(mu0, cov=Q0)
        y[n,0] = np.random.multivariate_normal(np.dot(C, u[n,0]), cov=R) 

        for t in range(1, T+1):
            u[n,t] = np.random.multivariate_normal(np.dot(A[t-1], u[n,t-1]), cov=Q[t-1])
            y[n,t] = np.random.multivariate_normal(np.dot(C, u[n,t]), cov=R)

    return u, y


def kalman_filter_loop(Y, A, C, Q, R, mu0, Q0):
    """ Kalman filter that broadcasts over the first dimension.
        
        Note: This function doesn't handle control inputs (yet).
        
        Y : ndarray, shape=(N, T+1, D)
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

    mu_predict = np.repeats(mu0, N, axis=0)
    sigma_predict = np.repeats(Q0, N, axis=0)

    mus_filt = np.empty((N, T+1, D))
    sigmas_filt = np.empty((N, T+1, D))

    ll = 0.

    for n in range(N):
        for t in range(T):

            # condition
            tmp1 = np.dot(C, sigma_predict)
            sigma_pred = np.dot(tmp1, C.T) + R
            L = np.linalg.cholesky(sigma_pred)
            v = solve_triangular(L, Y[n,t,:] - np.dot(C, mu))

            # log-likelihood over all trials
            ll += -0.5*np.dot(v,v) - np.sum(np.log(np.diag(L))) \
                  - T/2.*np.log(2.*np.pi)

            mus_filt[n,t] = mu_predict + np.dot(tmp1.T, solve_triangular(L, v, 'T'))

            tmp2 = solve_triangular(L, tmp1)
            sigmas_filt[n,t] = sigmas_predict - np.dot(tmp2.T, tmp2)

            # prediction
            mu_predict = np.dot(A[t], mus_filt[t])
            sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]

    return ll, mus_filt, sigmas_filt


def kalman_filter(Y, A, C, Q, R):
    """ Kalman filter that broadcasts over the first dimension.
        
        Note: This function doesn't handle control inputs (yet).
        
        Y : ndarray, shape=(N, T+1, D)
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
    p = C.shape[0]

    mu_predict = np.repeats(mu0, N, axis=0)
    sigma_predict = np.repeats(Q0, N, axis=0)

    mus_filt = np.empty((N, T+1, D))
    sigmas_filt = np.empty((N, T+1, D))

    ll = 0.

    for t in range(T):

        raise RuntimeError("make sure broadcasting works for all functions")
        # I think a bunch of the `dot`s need to be `einsum`s

        # condition
        tmp1 = np.dot(C, sigma_predict)
        sigma_pred = np.dot(tmp1, C.T) + R
        L = np.linalg.cholesky(sigma_pred)
        v = solve_triangular(L, Y[...,t,:] - np.dot(C, mu))

        # log-likelihood over all trials
        ll += np.sum(-0.5*np.dot(v,v) - np.sum(np.log(np.diag(L))) \
                     - T/2.*np.log(2.*np.pi))

        mus_filt[t] = mu_predict + np.dot(tmp1.T, solve_triangular(L, v, 'T'))

        tmp2 = solve_triangular(L, tmp1)
        sigmas_filt[t] = sigmas_predict - np.dot(tmp2.T, tmp2)

        # prediction
        mu_predict = np.dot(A[t], mus_filt[t])
        sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]

    return ll, mus_filt, sigmas_filt


if __name__ == "__main__":

    T = 100
    ntrials = 10
    A = np.array([[np.cos(20), np.sin(20)], [-np.sin(20), np.cos(20)]])
    A = np.repeats(A, T, axis=0)

    C = np.eye(2)

    Q0 = 0.2*np.eye(2)
    Q = np.repeats(0.2*np.eye(2), T, axis=0)

    R = 0.1*np.eye(2)

    mu0 = np.zeros(2)

    u, y = lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials)
