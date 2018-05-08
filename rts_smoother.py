import numpy as np

from autograd_linalg import solve_triangular


def sym(X):
    """
        X : a square matrix
        returns a symmetric matrix
    """
    return 0.5*(X + T_(X))


def T_(X):
    """
        X : some (2+)D matrix

        Swaps the last and second to last axis. e.g. A_{abcd...xyz} -> A_{abcd...xzy}. 
        If the matrix is 2D this is the transpose (hence the name :O). Well what happens if you
        are 2+ dimensions but first few dimension are like time steps or trial #'s? last 2 are the info
        you actually care about

        then transposes all 2D matricies across time/trial #!
    """
    return np.swapaxes(X, -1, -2)


def kalman_filter_basic(Y, A, C, Q, R, mu0, Q0):
    """ Kalman filter that broadcasts over the first dimension.
        
        Note: This function doesn't handle control inputs (yet).
        
        Y : ndarray, shape=(T, D)
          Observations

        A : ndarray, shape=(T, D, D)
          Time-varying dynamics matrices
        
        C : ndarray, shape=(D, D)
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
    
    T = Y.shape[0]
    D = A.shape[1]

    predict_mu = np.zeros((T + 1, D))
    predict_sigma = np.zeros((T + 1, D, D))

    measure_mu = np.zeros((T, D))
    measure_sigma = np.zeros((T, D, D))

    predict_mu[0] = mu0
    predict_sigma[0] = Q0

    for t in range(T):
        # 1 MEASUREMENT STEP
        # 1.1 Gain Matrix
        # 1.1.1 Calculate S_t (eq 18.36 in Murphy)
        S_t = np.dot(C, predict_sigma[t])
        S_t = np.dot(S_t, C.T) + R
        #S_t = sym(S_t) # correct for rounding errors?
        #print('me', S_t)

        # 1.2.3 Calculate gain matrix
        # main idea: want (sigma) (C)' (S_t)^-1
        # this is ((S_t)^-1 (C) (sigma))' 
        # note S_t and sigma are covariance matricies! For S_t see (eq 18.36)

        # we will solve for (S_t)^-1 (C) (sigma) and take the transpose
        # this let's us do the cool cholesky trick which is #faster
        
        # solve the system S_t x = (C)(sigma)
        C_sigma = np.dot(C, predict_sigma[t])

        # solve the system L L' x = (C)(sigma), S_t = L L'
        L = np.linalg.cholesky(S_t)

        # first solve L v = C sigma
        v = solve_triangular(L, C_sigma, lower=True)

        # then find x = (S_t)^-1 (C) (sigma_predict)!
        gain_matrix = T_(solve_triangular(L, v, trans='T', lower=True))

        # 1.2 Calculate residuals
        residual = Y[t] - np.dot(C, predict_mu[t])

        # 1.3 Update mu
        measure_mu[t] = predict_mu[t] + np.dot(gain_matrix, residual) 

        # 1.4 update sigma
        tmp = np.identity(D) - np.dot(gain_matrix, C)
        measure_sigma[t] = np.dot(tmp, predict_sigma[t])

        print ('mu %d:\n' %t, measure_mu[t])
        print()

        print ('sigma %d:\n' %t, measure_sigma[t])
        print()

        # 2 PREDICT STEP
        predict_mu[t + 1] = np.dot(A[t], measure_mu[t])
        predict_sigma[t + 1] = np.dot(np.dot(A[t], measure_sigma[t]), A[t].T) + Q[t]

    return measure_mu, measure_sigma


