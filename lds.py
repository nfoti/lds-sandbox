from __future__ import division
from __future__ import print_function

import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
plt.ion()

try:
    from autograd_linalg import solve_triangular
except ImportError:
    raise RuntimeError("must install `autograd_linalg` package")

# einsum2 is a parallel version of einsum that works for two arguments
try:
    from einsum2 import einsum2
except ImportError:
    # rename standard numpy function if don't have einsum2
    print("=> WARNING: using standard numpy.einsum,",
          "consider installing einsum2 package")
    from numpy import einsum as einsum2

from autograd.scipy.linalg import block_diag


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


def sym(X):
    """
        X : a square matrix
        returns a symmetric matrix
    """
    return 0.5*(X + T_(X))


def dot3(A, B, C):
    """
        A, B, C: all matricies (or scalars) that can be multiplled like ABC
        returns ABC where A, B, C are matricies/scalars
    """
    return np.dot(A, np.dot(B, C))


hs = lambda *args: np.concatenate(*args, axis=-1)
vs = lambda *args: np.concatenate(*args, axis=-2)
square = lambda X: np.dot(X, T_(X))
# rand_psd = lambda n: square(np.random.randn(n, n))


def rand_psd(n, minew=0.1, maxew=1.):
    """
        TODO : halp, what is this

        n : the dimensions of the random positive semi-definite matrix to return
        returns a positive semidefinite matrix that is presumbly random :O
    """

    X = np.random.randn(n,n)
    S = np.dot(T_(X), X)
    S = sym(S)
    ew, ev = np.linalg.eigh(S)
    ew -= np.min(ew)
    ew /= np.max(ew)
    ew *= (maxew - minew)
    ew += minew
    return dot3(ev, np.diag(ew), T_(ev))


def _ensure_ndim(X, T, ndim):
    """
        TODO
    """
    from numpy.lib.stride_tricks import as_strided as ast

    # essentially ensures underlying representation in memory of array
    # making copy if requried -> c-like contiguous rowiest continuity
    # doubles etc.
    X = np.require(X, dtype=np.float64, requirements='C')
    assert ndim-1 <= X.ndim <= ndim
    if X.ndim == ndim:
        assert X.shape[0] == T
        return X
    else:
        return ast(X, shape=(T,) + X.shape, strides=(0,) + X.strides)


def rand_stable(d, s=0.9):
    """
        TODO
    """
    A = np.random.randn(d, d)
    A *= s / np.max(np.abs(np.linalg.eigvals(A)))
    return A


def component_matrix(As, nlags):
    """ compute component form of latent VAR process
        
        [A_1 A_2 ... A_p]
        [ I   0  ...  0 ]
        [ 0   I   0   0 ]
        [ 0 ...   I   0 ]
    """

    d = As.shape[0]
    res = np.zeros((d*nlags, d*nlags))
    res[:d] = As
    
    if nlags > 1:
        res[np.arange(d,d*nlags), np.arange(d*nlags-d)] = 1

    return res


def lds_plot_progress(A_true, cur_A, Q_true, cur_Q, D, fig_quad, axes_quad, save=False, save_name=None):
    '''
    :param A_true: shape is (T, D, D)
    :param cur_A: shape is (T, D, D)
    '''
    for i in range(D):
        for j in range(D):
            axes_quad[i, j].cla()

            if A_true is not None:
                axes_quad[i, j].plot(A_true[:, i, j], color='green')
            if Q_true is not None:
                axes_quad[i, j].plot(Q_true[:, i, j], color='green', linestyle='--')
            if cur_A is not None:
                axes_quad[i, j].plot(cur_A[:, i, j], color='red')
            if cur_Q is not None:
                axes_quad[i, j].plot(cur_Q[:, i, j], color='red', linestyle='--')
            axes_quad[i, j].set_ylim(-1.5, 1.5)
    fig_quad.canvas.draw()

    if save and save_name is not None:
        fig_quad.savefig(str(save_name) + '.png')

    plt.pause(1.0 / 60.0)


def lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials):
    """ Simulates LDS with the following parameters. duh!
        Note: This function doesn't handle control inputs (yet).

        T : number of time steps per trial

        A : ndarray, shape=(T, d, d)
          State matrix. 
        
        C : ndarray, shape=(D, d)
          Observation matrix. 

        mu0: ndarray, shape=(d,)
          mean of initial state variable

        Q0 : ndarray, shape=(d, d)
          Covariance of initial state variable

        Q : ndarray, shape=(T, d, d)
          Covariance of latent states
        
        R : ndarray, shape=(T, D, D)
          Covariance of observations

        returns (x, y) where and x and y contain p(x, y)
    """
    # write version that broadcasts over trials at some point
    # d is the number of states, D is the number of outputs
    d = A.shape[1]
    D = C.shape[0]

    # notes does not give 0'd matricies
    # as if you just malloc'd!
    x = np.empty((ntrials, T, d))
    y = np.empty((ntrials, T, D))
 
    # This is a "trick" to generate samples which 
    # follow the covariance matricies efficiently
    # http://www.mathworks.com/help/matlab/ref/randn.html 
    # (under bivariate normal RVs)
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
        
        R : ndarray, shape=(T, p?, p?)
          Covariance of observations

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable
    """

    N = Y.shape[0]
    T, D, _ = A.shape
    p = C.shape[0]

    # result of prediction step, that is p(z_t | y_1:t-1)
    mu_predict = np.stack([mu0 for _ in range(N)], axis=0)
    sigma_predict = np.stack([Q0 for _ in range(N)], axis=0)

    # result of measurmeent step, that is p(z_t | y_1:t)
    mus_filt = np.zeros((N, T, D))
    sigmas_filt = np.zeros((N, T, D, D))

    ll = 0.

    for n in range(N):
        for t in range(T):

            # condition (measurement step)
            tmp1 = np.dot(C, sigma_predict[n])

            # corresponds to S_t matrix in murphy (eq 18.35)
            # S_t = cov [residuals for output pred. | past output pred.]
            sigma_pred = np.dot(tmp1, C.T) + R 

            L = np.linalg.cholesky(sigma_pred)

            # We want S_t^-1 r_t so solve for x in S_t^-1 x = r_t
            # Use Cholesky to get L L' x = r_t
            # solve for v = L' x (group it up)
            v = solve_triangular(L, Y[n,t,:] - np.dot(C, mu_predict[n]), lower=True)

            # log-likelihood over all trials
            ll += -0.5*np.dot(v,v) - 2.*np.sum(np.log(np.diag(L))) \
                  - 0.5*np.log(2.*np.pi)

            # solve_triangular(L, v, trans='T', lower=True) solves for the original x in L L' x = r_t
            # by solving the eq. v = L' x (group it up). Now we have x = S_t^-1 r_t
            # we note that we want (sigma_pred) (C') S_t^-1 r_t
            # tmp1 = C (sigma_pred) so tmp1' = (sigma_pred)' C' = (sigma_pred) C'
            # because covaraince matricies are symmetric! then boom we have what we want and it's just
            # eq. 18.31 in Murphy
            mus_filt[n,t,:] = mu_predict[n] + np.dot(tmp1.T, solve_triangular(L, v, trans='T', lower=True))

            tmp2 = solve_triangular(L, tmp1, lower=True)
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

    Qt = _ensure_ndim(Q, T, 3)

    mu_predict = np.stack([mu0 for _ in range(N)], axis=0)
    sigma_predict = np.stack([Q0 for _ in range(N)], axis=0)

    mus_filt = np.zeros((N, T, D))
    sigmas_filt = np.zeros((N, T, D, D))

    ll = 0.

    for t in range(T):
        '''
           # condition (measurement step)
            tmp1 = np.dot(C, sigma_predict[n])

            # corresponds to S_t matrix in murphy (eq 18.35)
            # S_t = cov [residuals for output pred. | past output pred.]
            sigma_pred = np.dot(tmp1, C.T) + R 

            L = np.linalg.cholesky(sigma_pred)

            # We want S_t^-1 r_t so solve for x in S_t^-1 x = r_t
            # Use Cholesky to get L L' x = r_t
            # solve for v = L' x (group it up)
            v = solve_triangular(L, Y[n,t,:] - np.dot(C, mu_predict[n]), lower=True)

            # log-likelihood over all trials
            ll += -0.5*np.dot(v,v) - 2.*np.sum(np.log(np.diag(L))) \
                  - 0.5*np.log(2.*np.pi)

            # solve_triangular(L, v, trans='T', lower=True) solves for the original x in L L' x = r_t
            # by solving the eq. v = L' x (group it up). Now we have x = S_t^-1 r_t
            # we note that we want (sigma_pred) (C') S_t^-1 r_t
            # tmp1 = C (sigma_pred) so tmp1' = (sigma_pred)' C' = (sigma_pred) C'
            # because covaraince matricies are symmetric! then boom we have what we want and it's just
            # eq. 18.31 in Murphy
            mus_filt[n,t,:] = mu_predict[n] + np.dot(tmp1.T, solve_triangular(L, v, trans='T', lower=True))

            tmp2 = solve_triangular(L, tmp1, lower=True)
            sigmas_filt[n,t,:,:] = sigma_predict[n] - np.dot(tmp2.T, tmp2)

            # prediction
            mu_predict[n] = np.dot(A[t], mus_filt[n,t,:])
            sigma_predict[n] = dot3(A[t], sigmas_filt[n,t,:,:], A[t].T) + Q[t]

        '''

        # condition
        # dot3(C, sigma_predict, C.T) + R
        #tmp1 = np.einsum('ik,nkj->nij', C, sigma_predict)
        tmp1 = einsum2('ik,nkj->nij', C, sigma_predict)
        sigma_pred = np.dot(tmp1, C.T) + R
        sigma_pred = sym(sigma_pred)

        L = np.linalg.cholesky(sigma_pred)
        # res[n] = Y[n,t,:] = np.dot(C, mu_predict[n])
        # the transpose works b/c of how dot broadcasts
        res = Y[...,t,:] - np.dot(mu_predict, C.T)
        v = solve_triangular(L, res, lower=True)
        
        # log-likelihood over all trials
        ll += (-0.5*np.sum(v*v)
               - 2.*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2))) 
               - N/2.*np.log(2.*np.pi))

        mus_filt[...,t,:] = mu_predict + einsum2('nki,nk->ni', tmp1, solve_triangular(L, v, trans='T', lower=True))

        tmp2 = solve_triangular(L, tmp1, lower=True)
        #sigmas_filt[...,t,:,:] = sigma_predict - np.einsum('nki,nkj->nij', tmp2, tmp2)
        sigmas_filt[...,t,:,:] = sym(sigma_predict - einsum2('nki,nkj->nij', tmp2, tmp2))

        # prediction
        #mu_predict = np.dot(A[t], mus_filt[t])
        #mu_predict = np.einsum('ik,nk->ni', A[t], mus_filt[...,t,:])
        mu_predict = einsum2('ik,nk->ni', A[t], mus_filt[...,t,:])

        #sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]
        #sigma_predict = np.einsum('ik,nkl,jl->nij', A[t], sigmas_filt[...,t,:,:], A[t]) + Q[t]
        tmp = einsum2('ik,nkl->nil', A[t], sigmas_filt[...,t,:,:])
        sigma_predict = sym(einsum2('nil,jl->nij', tmp, A[t]) + Qt[t])

    return ll, mus_filt, sigmas_filt


def rts_smooth_loop(Y, A, C, Q, R, mu0, Q0):

    N = Y.shape[0]
    T, D, _ = A.shape
    p = C.shape[0]

    mu_predict = np.zeros((T+1, D))
    sigma_predict = np.zeros((T+1, D, D))

    mus_smooth = np.empty((N, T, D))
    sigmas_smooth = np.empty((N, T, D, D))
    sigmas_smooth_tnt = np.empty((N, T-1, D, D))

    ll = 0.

    for n in range(N):
        mu_predict[0] = mu0
        sigma_predict[0] = Q0

        # just run the forwardpass like before
        for t in range(T):

            # condition
            tmp1 = np.dot(C, sigma_predict[t,:,:])
            sigma_pred = np.dot(tmp1, C.T) + R
            L = np.linalg.cholesky(sigma_pred)
            v = solve_triangular(L, Y[n,t,:] - np.dot(C, mu_predict[t,:]), lower=True)

            # log-likelihood over all trials
            ll += -0.5*np.dot(v,v) - 2.*np.sum(np.log(np.diag(L))) \
                  - 0.5*np.log(2.*np.pi)

            mus_smooth[n,t,:] = mu_predict[t] + np.dot(tmp1.T, solve_triangular(L, v, trans='T', lower=True))

            tmp2 = solve_triangular(L, tmp1, lower=True)
            sigmas_smooth[n,t,:,:] = sigma_predict[t] - np.dot(tmp2.T, tmp2)

            # prediction
            mu_predict[t+1] = np.dot(A[t], mus_smooth[n,t,:])
            sigma_predict[t+1] = dot3(A[t], sigmas_smooth[n,t,:,:], A[t].T) + Q[t]


        for t in range(T-2, -1, -1):
            
            # these names are stolen from mattjj and scott
            temp_nn = np.dot(A[t], sigmas_smooth[n,t,:,:])
            L = np.linalg.cholesky(sigma_predict[t+1,:,:])

            v = solve_triangular(L, temp_nn, lower=True)
            Gt_T = solve_triangular(L, v, trans='T', lower=True)

            # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
            # overwriting them on purpose
            mus_smooth[n,t,:] = mus_smooth[n,t,:] + np.dot(T_(Gt_T), mus_smooth[n,t+1,:] - mu_predict[t+1,:])
            sigmas_smooth[n,t,:,:] = sigmas_smooth[n,t,:,:] + dot3(T_(Gt_T), sigmas_smooth[n,t+1,:,:] - sigma_predict[:,t+1,:,:], Gt_T)
            sigmas_smooth_tnt[n,t,:,:] = np.dot(sigmas_smooth[n,t+1,:,:], Gt_T)

    return ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt


def rts_smooth(Y, A, C, Q, R, mu0, Q0):

    N = Y.shape[0]
    T, D, _ = A.shape
    p = C.shape[0]

    mu_predict = np.zeros((N, T+1, D))
    sigma_predict = np.zeros((N, T+1, D, D))
    mu_predict[:,0,:] = mu0
    sigma_predict[:,0,:,:] = Q0

    Qt = _ensure_ndim(Q, T, 3)

    mus_smooth = np.empty((N, T, D))
    sigmas_smooth = np.empty((N, T, D, D))
    sigmas_smooth_tnt = np.empty((N, T-1, D, D))

    ll = 0.

    for t in range(T):

        # condition
        # dot3(C, sigma_predict, C.T) + R
        #tmp1 = np.einsum('ik,nkj->nij', C, sigma_predict)
        tmp1 = einsum2('ik,nkj->nij', C, sigma_predict[:,t,:,:])
        sigma_pred = np.dot(tmp1, C.T) + R
        sigma_pred = sym(sigma_pred)

        L = np.linalg.cholesky(sigma_pred)
        # res[n] = Y[n,t,:] - np.dot(C, mu_predict[n])
        # the transpose works b/c of how dot broadcasts
        res = Y[...,t,:] - np.dot(mu_predict[:,t,:], C.T)
        v = solve_triangular(L, res, lower=True)
        
        # log-likelihood over all trials
        ll += (-0.5*np.sum(v*v)
               - 2.*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2))) 
               - p/2.*np.log(2.*np.pi))

        #mus_smooth[...,t,:] = mu_predict + np.einsum('nki,nk->ni', tmp1, solve_triangular(L, v, 'T'))
        mus_smooth[...,t,:] = mu_predict[:,t,:] + einsum2('nki,nk->ni',
                                                          tmp1,
                                                          solve_triangular(L, v, trans='T', lower=True))

        tmp2 = solve_triangular(L, tmp1, lower=True)
        #sigmas_smooth[...,t,:,:] = sigma_predict - np.einsum('nki,nkj->nij', tmp2, tmp2)
        sigmas_smooth[...,t,:,:] = sym(sigma_predict[:,t,:,:] - einsum2('nki,nkj->nij', tmp2, tmp2))

        # prediction
        #mu_predict = np.dot(A[t], mus_smooth[t])
        #mu_predict = np.einsum('ik,nk->ni', A[t], mus_smooth[...,t,:])
        mu_predict[:,t+1,:] = einsum2('ik,nk->ni', A[t], mus_smooth[...,t,:])

        #sigma_predict = dot3(A[t], sigmas_smooth[t], A[t].T) + Q[t]
        #sigma_predict = np.einsum('ik,nkl,jl->nij', A[t], sigmas_smooth[...,t,:,:], A[t]) + Q[t]
        tmp = einsum2('ik,nkl->nil', A[t], sigmas_smooth[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum2('nil,jl->nij', tmp, A[t]) + Qt[t])

    for t in range(T-2, -1, -1):

        # these names are stolen from mattjj and scott
        #temp_nn = np.dot(A[t], sigmas_smooth[n,t,:,:])
        temp_nn = einsum2('ik,nkj->nij', A[t], sigmas_smooth[:,t,:,:])
        L = np.linalg.cholesky(sigma_predict[:,t+1,:,:])

        v = solve_triangular(L, temp_nn, lower=True)
        # Look in Saarka for dfn of Gt_T
        Gt_T = solve_triangular(L, v, trans='T', lower=True)

        # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
        # overwriting them on purpose
        #mus_smooth[n,t,:] = mus_smooth[n,t,:] + np.dot(T_(Gt_T), mus_smooth[n,t+1,:] - mu_predict[t+1,:])
        mus_smooth[:,t,:] = mus_smooth[:,t,:] + einsum2('nki,nk->ni', Gt_T, mus_smooth[:,t+1,:] - mu_predict[:,t+1,:])

        tmp = einsum2('nki,nkj->nij', Gt_T, sigmas_smooth[:,t+1,:,:] - sigma_predict[:,t+1,:,:])
        tmp = einsum2('nik,nkj->nij', tmp, Gt_T)
        sigmas_smooth[:,t,:,:] = sym(sigmas_smooth[:,t,:,:] + tmp)

        # don't symmetrize this one
        #sigmas_smooth_tnt[n,t,:,:] = np.dot(sigmas_smooth[n,t+1,:,:], Gt_T)
        sigmas_smooth_tnt[:,t,:,:] = einsum2('nik,nkj->nij', sigmas_smooth[:,t+1,:,:], Gt_T)

    return ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt


def rts_smooth_fast(Y, A, C, Q, R, mu0, Q0, compute_lag1_cov=False):
    """ RTS smoother that broadcasts over the first dimension.
        Handles multiple lag dependence using component form.

        Note: This function doesn't handle control inputs (yet).

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D*nlag, D*nlag)
          Time-varying dynamics matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """

    N, T, _ = Y.shape
    _, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    L_R = np.linalg.cholesky(R)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])
    tmp = solve_triangular(L_R, CC, lower=True)
    Rinv_CC = solve_triangular(L_R, tmp, trans='T', lower=True)
    CCT_Rinv_CC = einsum2('ki,kj->ij', CC, Rinv_CC)

    # tile L_R across number of trials so solve_triangular
    # can broadcast over trials properly
    L_R = np.tile(L_R, (N, 1, 1))

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.empty((N, T+1, Dnlags))
    sigma_predict = np.empty((N, T+1, Dnlags, Dnlags))

    mus_smooth = np.empty((N, T, Dnlags))
    sigmas_smooth = np.empty((N, T, Dnlags, Dnlags))

    if compute_lag1_cov:
        sigmas_smooth_tnt = np.empty((N, T-1, Dnlags, Dnlags))
    else:
        sigmas_smooth_tnt = None

    ll = 0.
    mu_predict[:,0,:] = np.tile(mu0, nlags)
    sigma_predict[:,0,:,:] = QQ0.copy()

    I_tiled = np.tile(np.eye(D), (N, 1, 1))

    for t in range(T):

        # condition
        # sigma_x = dot3(C, sigma_predict, C.T) + R
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict[:,t,:,:])

        res = Y[...,t,:] - einsum2('ik,nk->ni', CC, mu_predict[...,t,:])

        # Rinv * res
        tmp2 = solve_triangular(L_R, res, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * res
        tmp3 = einsum2('ki,nk->ni', Rinv_CC, res)

        # (Pinv + C^T Rinv C)_inv * tmp3
        # Pinv = np.linalg.inv(sigma_predict[:,t,:,:])
        L_P = np.linalg.cholesky(sigma_predict[:,t,:,:])
        tmp = solve_triangular(L_P, I_tiled, lower=True)
        Pinv = solve_triangular(L_P, tmp, trans='T', lower=True)
        tmp4 = sym(Pinv + CCT_Rinv_CC)
        L_tmp4 = np.linalg.cholesky(tmp4)
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)
        
        # Rinv C * tmp3
        tmp3 = einsum2('ik,nk->ni', Rinv_CC, tmp3)

        # add the two Woodbury * res terms together
        tmp = tmp2 - tmp3
        
        # # log-likelihood over all trials
        # # TODO: recompute with new tmp variables
        # ll += (-0.5*np.sum(v*v)
        #        - 2.*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)))
        #        - p/2.*np.log(2.*np.pi))

        mus_smooth[:,t,:] = mu_predict[:,t,:] + einsum2('nki,nk->ni', tmp1, tmp)

        # tmp2 = L^{-1}*C*sigma_predict
        # tmp2 = solve_triangular(L, tmp1, lower=True)

        # Rinv * tmp1
        tmp2 = solve_triangular(L_R, tmp1, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * tmp1
        tmp3 = einsum2('ki,nkj->nij', Rinv_CC, tmp1)

        # (Pinv + C^T Rinv C)_inv * tmp3
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)
        
        # Rinv C * tmp3
        tmp3 = einsum2('ik,nkj->nij', Rinv_CC, tmp3)

        # add the two Woodbury * tmp1 terms together, left-multiply by tmp1
        tmp = einsum2('nki,nkj->nij', tmp1, tmp2 - tmp3)

        sigmas_smooth[:,t,:,:] = sym(sigma_predict[:,t,:,:] - tmp)

        # prediction
        #mu_predict = np.dot(A[t], mus_smooth[t])
        mu_predict[:,t+1,:] = einsum2('ik,nk->ni', AA[t], mus_smooth[:,t,:])

        #sigma_predict = dot3(A[t], sigmas_smooth[t], A[t].T) + Q[t]
        tmp = einsum2('ik,nkl->nil', AA[t], sigmas_smooth[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum2('nil,jl->nij', tmp, AA[t]) + QQ[t])
            
    for t in range(T-2, -1, -1):
        
        # these names are stolen from mattjj and slinderman
        #temp_nn = np.dot(A[t], sigmas_smooth[n,t,:,:])
        temp_nn = einsum2('ik,nkj->nij', AA[t], sigmas_smooth[:,t,:,:])

        L = np.linalg.cholesky(sigma_predict[:,t+1,:,:])
        v = solve_triangular(L, temp_nn, lower=True)
        # Look in Saarka for dfn of Gt_T
        Gt_T = solve_triangular(L, v, trans='T', lower=True)

        # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
        # overwriting them on purpose
        #mus_smooth[n,t,:] = mus_smooth[n,t,:] + np.dot(T_(Gt_T), mus_smooth[n,t+1,:] - mu_predict[t+1,:])
        mus_smooth[:,t,:] = mus_smooth[:,t,:] + einsum2('nki,nk->ni', Gt_T, mus_smooth[:,t+1,:] - mu_predict[:,t+1,:])

        #sigmas_smooth[n,t,:,:] = sigmas_smooth[n,t,:,:] + dot3(T_(Gt_T), sigmas_smooth[n,t+1,:,:] - temp_nn, Gt_T)
        tmp = einsum2('nki,nkj->nij', Gt_T, sigmas_smooth[:,t+1,:,:] - sigma_predict[:,t+1,:,:])
        tmp = einsum2('nik,nkj->nij', tmp, Gt_T)
        sigmas_smooth[:,t,:,:] = sym(sigmas_smooth[:,t,:,:] + tmp)

        if compute_lag1_cov:
            # This matrix is NOT symmetric, so don't symmetrize!
            #sigmas_smooth_tnt[n,t,:,:] = np.dot(sigmas_smooth[n,t+1,:,:], Gt_T)
            sigmas_smooth_tnt[:,t,:,:] = einsum2('nik,nkj->nij', sigmas_smooth[:,t+1,:,:], Gt_T)

    return ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt


# what was this
def em_objective(Y, D, params, fixedparams, ldsregparams,
                 mus_smooth, sigmas_smooth, sigmas_tnt_smooth):

    At, L_Q, L_Q0 = params

    ntrials, T, p = Y.shape

    C, L_R = fixedparams

    w_s = 1.
    x_smooth_0_outer = einsum2('ri,rj->rij', mus_smooth[:,0,:D],
                                             mus_smooth[:,0,:D])
    B0 = w_s*np.sum(sigmas_smooth[:,0,:D,:D] + x_smooth_0_outer,
                    axis=0)

    x_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,1:,:D],
                                              mus_smooth[:,1:,:D])
    B1 = w_s*np.sum(sigmas_smooth[:,1:,:D,:D] + x_smooth_outer, axis=0)

    z_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,:-1,:],
                                              mus_smooth[:,:-1,:])
    B3 = w_s*np.sum(sigmas_smooth[:,:-1,:,:] + z_smooth_outer, axis=0)

    # this was the original
    #B1_B3 = w_s*np.sum(sigmas_smooth + mus_smooth_outer, axis=0)
    #B1, B3 = B1_B3[1:], B1_B3[:-1]

    mus_smooth_outer_l1 = einsum2('rti,rtj->rtij',
                                  mus_smooth[:,1:,:D],
                                  mus_smooth[:,:-1,:])
    B2 = w_s*np.sum(sigmas_tnt_smooth[:,:,:D,:] + mus_smooth_outer_l1, axis=0)

    L_Q0_inv_B0 = solve_triangular(L_Q0, B0, lower=True)
    L1 = (ntrials*2.*np.sum(np.log(np.diag(L_Q0)))
            + np.trace(solve_triangular(L_Q0, L_Q0_inv_B0, lower=True, trans='T')))

    AtB2T = einsum2('tik,tjk->tij', At, B2)
    B2AtT = einsum2('tik,tjk->tij', B2, At)
    tmp = einsum2('tik,tkl->til', At, B3)
    AtB3AtT = einsum2('tik,tjk->tij', tmp, At)

    tmp = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)

    L_Q_inv_tmp = solve_triangular(L_Q, tmp, lower=True)
    L2 = (ntrials*(T-1)*2.*np.sum(np.log(np.diag(L_Q)))
            + np.trace(solve_triangular(L_Q, L_Q_inv_tmp, lower=True, trans='T')))

    lam0, lam1 = ldsregparams
    penalty = 0.
    if lam0 > 0.:
        penalty += lam0*np.sum(At**2)
    if lam1 > 0.:
        AtmAtm1_2 = (At[1:] - At[:-1])**2
        penalty += lam1*np.sum(AtmAtm1_2)

    res = Y - einsum2('ik,ntk->nti', C, mus_smooth[:,:,:D])
    CP_smooth = einsum2('ik,ntkj->ntij', C, sigmas_smooth[:,:,:D,:D])
    B4 = w_s*(np.sum(einsum2('nti,ntj->ntij', res, res), axis=(0,1))
              + np.sum(einsum2('ntik,jk->ntij', CP_smooth, C),
                       axis=(0,1)))
    L_R_inv_B4 = solve_triangular(L_R, B4, lower=True)
    L3 = (ntrials*T*2.*np.sum(np.log(np.diag(L_R)))
            + np.trace(solve_triangular(L_R, L_R_inv_B4, lower=True, trans='T')))

    return L1 + L2 + L3 + penalty, L1, L2, L3, penalty


def em(Y, initparams, fixedparams, ldsregparams, niter=10, Atrue=None, plot_progress=False, save_plots=False, debug=False, num_objvals=5, tol=1e-6):
    A_init, Q_init, Q0_init = initparams
    A = A_init.copy()
    Q = Q_init.copy()
    Q0 = Q0_init.copy()
    
    L_Q = np.linalg.cholesky(Q)

    _, D, Dnlags = A.shape
    nlags = Dnlags // D
    ntrials, T, p = Y.shape

    C, R, mu0 = fixedparams

    L_R = np.linalg.cholesky(R)

    lam0, lam1 = ldsregparams
    
    em_obj_list = np.zeros(niter)
    best_em_obj = np.finfo('float').max
    bestparams = (A.copy(), Q.copy(), Q0.copy())

    At = A[:-1]

    fig_quad, axes_quad = None, None
    if plot_progress:
        fig_quad, axes_quad = plt.subplots(D, D, figsize=(12, 6))
        lds_plot_progress(Atrue, At, None, None, D, fig_quad, axes_quad,
                          save=save_plots, save_name='fullem_iteration0')

    for em_it in range(niter):
        print('Cur it', em_it)
        L_Q0 = np.linalg.cholesky(Q0)

        # why repeat this line
        Q = np.dot(L_Q, L_Q.T)

        # e-step
        smoothed_state_params = rts_smooth(Y, A, C, Q, R, mu0, Q0)
        ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = smoothed_state_params
        #print(ll)
        w_s = 1.
        x_smooth_0_outer = einsum2('ri,rj->rij', mus_smooth[:,0,:D],
                                                 mus_smooth[:,0,:D])
        B0 = w_s*np.sum(sigmas_smooth[:,0,:D,:D] + x_smooth_0_outer,
                        axis=0)
        
        x_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,1:,:D],
                                                  mus_smooth[:,1:,:D])
        B1 = w_s*np.sum(sigmas_smooth[:,1:,:D,:D] + x_smooth_outer, axis=0)

        z_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,:-1,:],
                                                  mus_smooth[:,:-1,:])
        B3 = w_s*np.sum(sigmas_smooth[:,:-1,:,:] + z_smooth_outer, axis=0)

        # this was the original
        #B1_B3 = w_s*np.sum(sigmas_smooth + mus_smooth_outer, axis=0)
        #B1, B3 = B1_B3[1:], B1_B3[:-1]

        mus_smooth_outer_l1 = einsum2('rti,rtj->rtij',
                                      mus_smooth[:,1:,:D],
                                      mus_smooth[:,:-1,:])
        B2 = w_s*np.sum(sigmas_smooth_tnt[:,:,:D,:] + mus_smooth_outer_l1, axis=0)

        if debug:
            it_params = (At, L_Q, Q0)
            it_fixedparams = (C, L_R)
            em_obj, L1, L2, L3, penalty = em_objective(Y, D, it_params, it_fixedparams, ldsregparams,
                                  mus_smooth, sigmas_smooth, sigmas_smooth_tnt)
            em_obj_list[em_it] = em_obj

            # check for updated best iterate
            if em_obj < best_em_obj:
                best_em_obj = em_obj
                bestparams = (A.copy(), Q.copy(), Q0.copy())

            # check for convergence
            if em_it >= num_objvals:
                vals_to_check = em_obj_list[em_it-num_objvals:em_it]
                if np.all(np.abs((vals_to_check - em_obj) / em_obj) <= tol):
                    print('EM objective converged')
                    em_obj_list = em_obj_list[:em_it+1]
                    break

            print('em iter:', em_it+1, 'EM objective: ', em_obj)
            print('  L1:', L1)
            print('  L2:', L2)
            print('  L3:', L3)
            print('  pen:', penalty)
            print(np.dot(L_Q, L_Q.T))

        # m-step
        Q0 = 1./(ntrials) * B0

        # joint objective for At and L_Q
        def L2_obj(At, L_Q):

            AtB2T = einsum2('tik,tjk->tij', At, B2)
            B2AtT = einsum2('tik,tjk->tij', B2, At)
            # einsum2 is faster
            #AtB3AtT = np.einsum('tik,tkl,tjl->tij', At, B3, At)
            tmp = einsum2('tik,tkl->til', At, B3)
            AtB3AtT = einsum2('til,tjl->tij', tmp, At)
            elbo_2 = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)

            L_Q_inv_elbo_2 = solve_triangular(L_Q, elbo_2, lower=True)
            obj = np.trace(solve_triangular(L_Q, L_Q_inv_elbo_2, lower=True, trans='T'))
            obj += lam0*np.sum(At**2)
            AtmAtm1_2 = (At[1:] - At[:-1])**2
            obj += lam1*np.sum(AtmAtm1_2)
            return obj

        # gradient descent
        grad_fun = grad(lambda At: L2_obj(At, L_Q))
        obj_diff = np.finfo('float').max
        obj = L2_obj(At, L_Q)
        it = 0
        maxiter = 100
        while np.abs(obj_diff / obj) > 1e-6:

            if it > maxiter:
                break

            obj_start = L2_obj(At, L_Q)
            grad_A_un = grad_fun(At)
            grad_A = grad_A_un # / np.linalg.norm(grad_A_un.flatten())
            tmp_diff = np.inf
            step_size = 1.
            tau = 0.8
            while tmp_diff > 0:
                new_At = At - step_size * grad_A
                ref = obj_start - step_size/10000. * (np.sum(grad_A_un*grad_A))
                obj = L2_obj(new_At, L_Q)
                tmp_diff = obj - ref
                step_size *= tau
            At[:] = new_At

            # update Q using closed form
            AtB2T = einsum2('tik,tjk->tij', At, B2)
            B2AtT = einsum2('tik,tjk->tij', B2, At)
            # einsum2 is faster
            #AtB3AtT = np.einsum('tik,tkl,tjl->tij', At, B3, At)
            tmp = einsum2('tik,tkl->til', At, B3)
            AtB3AtT = einsum2('til,tjl->tij', tmp, At)
            elbo_2 = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)
            Q = 1./(ntrials*T) * elbo_2
            L_Q = np.linalg.cholesky(Q)

            obj = L2_obj(At, L_Q)
            obj_diff = obj_start - obj
            it += 1

            #print('took', it, 'iters\t', 'step-size:', step_size / tau)

            #if it % 10 == 0:
            #    # plot true and estimated dynamics for each (i, j) entry
            #    for i in range(D):
            #        for j in range(D):
            #            axes_quad[i,j].cla()
            #            if Atrue is not None:
            #                axes_quad[i,j].plot(Atrue[:-1, i, j], color='green')
            #            axes_quad[i,j].plot(At[:, i, j], color='red')
            #            axes_quad[i,j].set_ylim(-1.5, 1.5)
            #    fig_quad.canvas.draw()
            #    plt.pause(1./60.)

        # plot true and estimated dynamics for each (i, j) entry
        if plot_progress:
            lds_plot_progress(Atrue, At, None, None, D, fig_quad, axes_quad,
                              save=save_plots, save_name='fullem_iteration' + str(em_it))

        ## plot function curves in direction of gradient for At and L_Q
        #interp_vals = np.linspace(-1, 1, 101)
        #objvals = [At_obj((At + t*grad_A, L_Q)) for t in interp_vals]
        #axes_inner[0].cla()
        #axes_inner[0].plot(interp_vals, objvals, color='blue')
        #axes_inner[0].scatter(0, At_obj((At, L_Q)), color='blue')
        #axes_inner[0].scatter(-step_size, At_obj((At - step_size*grad_A, L_Q)), color='red')
        #objvals = [At_obj((At, L_Q + t*grad_L_Q)) for t in interp_vals]
        #axes_inner[1].cla()
        #axes_inner[1].plot(interp_vals, objvals, color='blue')
        #axes_inner[1].scatter(0, At_obj((At, L_Q)), color='blue')
        #axes_inner[1].scatter(-step_size, At_obj((At, L_Q - step_size*grad_L_Q)), color='red')
        #fig_inner.canvas.draw()

    # retrieve best parameters
    A, Q, Q0 =  bestparams

    print('final smoothing with estimated parameters')
    _, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = rts_smooth(Y, A, C, Q, R, mu0, Q0)

    ret = {'A': A, 'L_Q': L_Q, 'L_Q0': L_Q0, 'mus_smooth': mus_smooth,
           'sigmas_smooth': sigmas_smooth, 'sigmas_smooth_tnt': sigmas_smooth_tnt,
           'em_obj_vals': em_obj_list}

    return ret



if __name__ == "__main__":

    #np.random.seed(8675309)
    np.random.seed(42)

    T = 165
    ntrials = 20
    #theta = 1.2
    #A = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    # the same for convenience. constructing reasonable C from small to large
    # dimensions is tricky

    d = 10
    D = 120

    #d = 05
    #D = 6

    # A = rand_stable(d)
    # A = np.stack([A for _ in range(T)], axis=0)
    A = np.array([0.5*np.eye(d) for _ in range(T)])
    f01 = np.sin(np.linspace(0., 2*np.pi, num=T))
    f10 = -np.sin(np.linspace(0., 2*np.pi, num=T) + 1.2)*f01
    A[:,0,1] = f01*np.random.randn()*np.sign(np.random.randn())
    A[:,1,0] = f10*np.random.randn()*np.sign(np.random.randn())
    A[-1] = np.zeros((d, d))


    #C = np.eye(D)
    C, _ = np.linalg.qr(np.random.randn(D, d))

    #Q0 = 0.5*np.eye(d)
    #Q = 0.5*np.eye(d)
    # Q = np.diag([2.9, 3.5, 3., 10.])
    Q0 = rand_psd(d, maxew=0.5)
    Q = rand_psd(d, maxew=0.5)
    Q_stack = np.stack([Q for _ in range(T)], axis=0)

    #R = 0.1*np.eye(D)
    #R[np.triu_indices_from(R, k=1)] = 1e-4
    #R[np.triu_indices_from(R, k=2)] = 0.
    #R[np.tril_indices_from(R, k=-1)] = 1e-4
    #R[np.tril_indices_from(R, k=-2)] = 0.

    R = rand_psd(D)

    mu0 = np.zeros(d)

    x, Y = lds_simulate_loop(T, A, C, Q_stack, R, mu0, Q0, ntrials)

    #A_fixed = rand_stable(d)
    #A_init = np.array([A_fixed for _ in range(T)])  # np.random.randn(*A.shape)
    A_init = np.array([rand_stable(d, s=0.7) for _ in range(T)])  # np.random.randn(*A.shape)
    # A_init = A.copy()

    #Q_init = Q.copy()
    Q_init = rand_psd(d)
    #Q_init = np.eye(d)

    # Q0_init = Q0.copy()
    Q0_init = rand_psd(d)
    #Q0_init = np.diag(np.random.rand(d))

    initparams = (A_init, Q_init, Q0_init)
    fixedparams = (C, R, mu0)
    ldsregparams = (0., 0.1)

    _, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = \
            rts_smooth(Y, A, C, Q, R, mu0, Q0)
    _, mus_smooth_fast, sigmas_smooth_fast, sigmas_smooth_tnt_fast = \
            rts_smooth_fast(Y, A, C, Q, R, mu0, Q0, compute_lag1_cov=True)
    assert np.allclose(mus_smooth, mus_smooth_fast), "mus don't match"
    assert np.allclose(sigmas_smooth, sigmas_smooth_fast), "sigmas don't match"
    assert np.allclose(sigmas_smooth_tnt, sigmas_smooth_tnt_fast), "sigmas_tnt don't match"

    ret = em(Y, initparams, fixedparams, ldsregparams, niter=50, Atrue=A)
    A_est = ret['A']
    em_obj_vals = ret['em_obj_vals']
    mus_smooth = ret['mus_smooth']
    Q_est = np.dot(ret['L_Q'], ret['L_Q'].T)
    Q0_est = np.dot(ret['L_Q0'], ret['L_Q0'].T)

    mus_smooth = ret['mus_smooth']
    _, mus_smooth_true, _, _ = rts_smooth(Y, A, C, Q, R, mu0, Q0)

    plt.figure()
    for i in range(d):
        mean_true = np.mean(x[:, :, i], axis=0)
        mean_smoothed = np.mean(mus_smooth[:, :, i], axis=0)
        mean_smoothed_true = np.mean(mus_smooth_true[:, :, i], axis=0)
        plt.subplot(d, 1, i + 1)
        plt.plot(mean_true, color='green', label='true (mean over trials)')
        plt.plot(mean_smoothed, color='red', label=r'smoothed with $A_{est}$')
        plt.plot(mean_smoothed_true, color='blue', label=r'smoothed with $A_{true}$')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(em_obj_vals)

    #print("running kalman filter tests")
    #ll_loop, mus_filt_loop, sigmas_filt_loop = kalman_filter_loop(Y, A, C, Q, R, mu0, Q0)
    #ll_kf, mus_filt, sigmas_filt = kalman_filter(Y, A, C, Q, R, mu0, Q0)

    #assert np.allclose(ll, ll_loop)
    #assert np.allclose(mus_filt, mus_filt_loop)
    #assert np.allclose(sigmas_filt, sigmas_filt_loop)

    #print("running rts tests")
    #ll1, mus_smooth1, sigmas_smooth1, sigmas_smooth_tnt1 = rts_smooth_loop(Y, A, C, Q, R, mu0, Q0)
    #ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = rts_smooth(Y, A, C, Q, R, mu0, Q0)

    #assert np.allclose(ll, ll1)
    #assert np.allclose(mus_smooth, mus_smooth1)
    #assert np.allclose(sigmas_smooth, sigmas_smooth1)
    #assert np.allclose(sigmas_smooth_tnt, sigmas_smooth_tnt1)
