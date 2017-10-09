from __future__ import division
from __future__ import print_function

import numpy as np

from scipy.linalg import block_diag

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


def rand_stable(d, s=0.9):
    A = np.random.randn(d, d)
    A *= 0.95 / np.max(np.abs(np.linalg.eigvals(A)))
    return A

def T_(X):
    return np.swapaxes(X, -1, -2)

def sym(X):
    return 0.5*(X + T_(X))

hs = lambda *args: np.concatenate(*args, axis=-1)
vs = lambda *args: np.concatenate(*args, axis=-2)

def component_matrix(As, nlags):
    """ compute component form of latent VAR process
        
        [A_1 A_2 ... A_p]
        [ 0   I  ...  0 ]
        [...      I   0 ]
        [ 0   0   0   0 ]

    """

    d = As.shape[0]
    res = np.zeros((d*nlags, d*nlags))
    res[:d] = As
    
    if nlags > 1:
        res[np.arange(d,d*nlags), np.arange(d*nlags-d)] = 1

    return res


def lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials):
    T, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])

    QQ = np.zeros((T, D*nlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    x = np.zeros((ntrials, T, Dnlags))
    y = np.zeros((ntrials, T, p))

    L_R = np.linalg.cholesky(R)
    L_Q = np.linalg.cholesky(Q)

    L_QQ = np.zeros((T,Dnlags, Dnlags))
    L_QQ[:,:D,:D] = L_Q

    mu0_bar = np.tile(mu0, nlags)
    
    for n in range(ntrials):
        x[n,0] = np.random.multivariate_normal(mu0_bar, cov=QQ0)
        y[n,0] = np.dot(CC, x[n,0]) + np.dot(L_R, np.random.randn(p))

        for t in range(1, T):
            x[n,t] = np.dot(AA[t-1], x[n,t-1]) + np.dot(L_QQ[t-1], np.random.randn(Dnlags))
            y[n,t] = np.dot(CC, x[n,t]) + np.dot(L_R, np.random.randn(p))

    z = x.copy()
    x = np.ascontiguousarray(x[:,:,:D])

    return x, y, z


def kalman_filter(Y, A, C, Q, R, mu0, Q0):
    
    N = Y.shape[0]
    T, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.stack([np.tile(mu0, nlags) for _ in range(N)], axis=0)
    sigma_predict = np.stack([QQ0 for _ in range(N)], axis=0)

    mus_filt = np.zeros((N, T, Dnlags))
    sigmas_filt = np.zeros((N, T, Dnlags, Dnlags))

    ll = 0.

    for t in range(T):

        # condition
        # dot3(CC, sigma_predict, CC.T) + R
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict)
        sigma_pred = np.dot(tmp1, CC.T) + R
        sigma_pred = sym(sigma_pred)

        res = Y[...,t,:] - np.dot(mu_predict, CC.T)

        L = np.linalg.cholesky(sigma_pred)
        v = solve_triangular(L, res, lower=True)

        # log-likelihood over all trials
        ll += (-0.5*np.sum(v*v)
               - np.sum(np.log(np.diagonal(L, axis1=1, axis2=2))) 
               - N/2.*np.log(2.*np.pi))

        mus_filt[...,t,:] = mu_predict + einsum2('nki,nk->ni', tmp1,
                                                               solve_triangular(L, v, 'T', lower=True))

        tmp2 = solve_triangular(L, tmp1, lower=True)
        sigmas_filt[...,t,:,:] = sym(sigma_predict - einsum2('nki,nkj->nij', tmp2, tmp2))

        # prediction
        mu_predict = einsum2('ik,nk->ni', AA[t], mus_filt[...,t,:])

        sigma_predict = einsum2('ik,nkl->nil', AA[t], sigmas_filt[...,t,:,:])
        sigma_predict = sym(einsum2('nil,jl->nij', sigma_predict, AA[t]) + QQ[t])

    return ll, mus_filt, sigmas_filt


def rts_smooth(Y, A, C, Q, R, mu0, Q0, compute_lag1_cov=False):

    N, T, _ = Y.shape
    _, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.empty((N, T+1, Dnlags))
    sigma_predict = np.empty((N, T+1, Dnlags, Dnlags))

    #mus_filt = np.zeros((N, T, Dnlags))
    #sigmas_filt = np.zeros((N, T, Dnlags, Dnlags))

    mus_smooth = np.empty((N, T, Dnlags))
    sigmas_smooth = np.empty((N, T, Dnlags, Dnlags))

    if compute_lag1_cov:
        sigmas_smooth_tnt = np.empty((N, T-1, Dnlags, Dnlags))
    else:
        sigmas_smooth_tnt = None

    ll = 0.
    mu_predict[:,0,:] = np.tile(mu0, nlags)
    sigma_predict[:,0,:,:] = QQ0.copy()

    for t in range(T):

        # condition
        # sigma_x = dot3(C, sigma_predict, C.T) + R
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict[:,t,:,:])
        sigma_x = einsum2('nik,jk->nij', tmp1, CC) + R
        sigma_x = sym(sigma_x)

        L = np.linalg.cholesky(sigma_x)
        # res[n] = Y[n,t,:] = np.dot(C, mu_predict[n,t,:])
        res = Y[...,t,:] - einsum2('ik,nk->ni', CC, mu_predict[...,t,:])
        v = solve_triangular(L, res, lower=True)
        
        # log-likelihood over all trials
        ll += (-0.5*np.sum(v*v)
               - np.sum(np.log(np.diagonal(L, axis1=1, axis2=2))) 
               - N/2.*np.log(2.*np.pi))

        mus_smooth[:,t,:] = mu_predict[:,t,:] + einsum2('nki,nk->ni',
                                                        tmp1,
                                                        solve_triangular(L, v, 'T', lower=True))

        # tmp2 = L^{-1}*C*sigma_predict
        tmp2 = solve_triangular(L, tmp1, lower=True)
        sigmas_smooth[:,t,:,:] = sym(sigma_predict[:,t,:,:] - einsum2('nki,nkj->nij', tmp2, tmp2))

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
        Gt_T = solve_triangular(L, v, 'T', lower=True)

        # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
        # overwriting them on purpose
        #mus_smooth[n,t,:] = mus_smooth[n,t,:] + np.dot(T_(Gt_T), mus_smooth[n,t+1,:] - mu_predict[t+1,:])
        mus_smooth[:,t,:] = mus_smooth[:,t,:] + einsum2('nki,nk->ni', Gt_T, mus_smooth[:,t+1,:] - mu_predict[:,t+1,:])

        #sigmas_smooth[n,t,:,:] = sigmas_smooth[n,t,:,:] + dot3(T_(Gt_T), sigmas_smooth[n,t+1,:,:] - temp_nn, Gt_T)
        tmp = einsum2('nki,nkj->nij', Gt_T, sigmas_smooth[:,t+1,:,:] - temp_nn)
        tmp = einsum2('nik,nkj->nij', tmp, Gt_T)
        sigmas_smooth[:,t,:,:] = sym(sigmas_smooth[:,t,:,:] + tmp)

        if compute_lag1_cov:
            # This matrix is NOT symmetric, so don't symmetrize!
            #sigmas_smooth_tnt[n,t,:,:] = np.dot(sigmas_smooth[n,t+1,:,:], Gt_T)
            sigmas_smooth_tnt[:,t,:,:] = einsum2('nik,nkj->nij', sigmas_smooth[:,t+1,:,:], Gt_T)

    return ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt


if __name__ == "__main__":

    #np.random.seed(8675309)
    np.random.seed(42)

    T = 165
    ntrials = 10
    #theta = 1.2
    #A = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    # the same for convenience. constructing reasonable C from small to large
    # dimensions is tricky

    AR_coef = np.array([[0.5965, 0.45, 0, 0.4250],
                        [0, 0.6364, 0, 0],
                        [0, 0.475, 0.4536, 0.4],
                        [0, 0, 0, 0.3536]])

    AR_coef_true = hs([AR_coef, -AR_coef, AR_coef, -AR_coef])
    d = AR_coef.shape[0]

    #d = 4
    #AR_coef_true = rand_stable(d)

    A = np.stack([AR_coef_true for _ in range(T)], axis=0)

    D = 4
    C = np.eye(D)
    #C = np.random.randn(D, d)

    Q0 = 0.2*np.eye(d)

    Q_true = np.diag([0.2, 0.6, 0.2, 0.6])
    Q = np.stack([Q_true for _ in range(T)], axis=0)

    R = 0.1*np.eye(D)

    mu0 = np.zeros(d)

    x, Y, z = lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials)

    print("kalman filter...")
    ll, mus_filt, sigmas_filt = kalman_filter(Y, A, C, Q, R, mu0, Q0)

    def plot_states_and_est(x, mus_filt, trial=0, mus_smooth=None):
        import matplotlib.pyplot as plt
        N, T, D = x.shape
        Dnlag = mus_filt.shape[-1]
        nlag = Dnlag // D

        fig, axes = plt.subplots(D, 1)
        for i, ax in enumerate(axes):
            ax.plot(x[trial,:,i], color='k', linestyle='dashed')
            ax.plot(mus_filt[trial,:,i])
            if mus_smooth is not None:
                ax.plot(mus_smooth[trial,:,i])
        plt.show()

    plot_states_and_est(x, mus_filt, trial=0)

    print("rts smoother...")
    ll_rts, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = \
            rts_smooth(Y, A, C, Q, R, mu0, Q0, compute_lag1_cov=True)
            
    import matplotlib.pyplot as plt
    for i in range(ntrials):
        plt.close()
        plot_states_and_est(x, mus_filt, trial=i, mus_smooth=mus_smooth)
        plt.pause(1.)

    #print("running kalman filter tests")
    ##ll_loop, mus_filt_loop, sigmas_filt_loop = kalman_filter_loop(Y, A, C, Q, R, mu0, Q0)
    ##ll_kf, mus_filt, sigmas_filt = kalman_filter(Y, A, C, Q, R, mu0, Q0)

    ##assert np.allclose(ll, ll_loop)
    ##assert np.allclose(mus_filt, mus_filt_loop)
    ##assert np.allclose(sigmas_filt, sigmas_filt_loop)

    #print("running rts tests")
    #ll1, mus_smooth1, sigmas_smooth1, sigmas_smooth_tnt1 = rts_smooth_loop(Y, A, C, Q, R, mu0, Q0)
    #ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = rts_smooth(Y, A, C, Q, R, mu0, Q0)

    #assert np.allclose(ll, ll1)
    #assert np.allclose(mus_smooth, mus_smooth1)
    #assert np.allclose(sigmas_smooth, sigmas_smooth1)
    #assert np.allclose(sigmas_smooth_tnt, sigmas_smooth_tnt1)

