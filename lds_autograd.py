from __future__ import division
from __future__ import print_function

import autograd.numpy as np
import matplotlib.pyplot as plt

from autograd import grad, make_hvp
from autograd.misc import flatten
from autograd.misc.optimizers import adam, sgd

from scipy.linalg import LinAlgError
from scipy.optimize import minimize


try:
    from autograd_linalg import solve_triangular
except ImportError:
    raise RuntimeError("must install `autograd_linalg` package")

try:
    from einsum2 import einsum2
except ImportError:
    # rename standard numpy function if don't have einsum2
    raise RuntimeError("must install `einsum2` package")


def T_(X):
    return np.swapaxes(X, -1, -2)

def sym(X):
    return 0.5*(X + T_(X))

def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))

square = lambda X: np.dot(X, T_(X))
rand_psd = lambda n: square(np.random.randn(n, n))


def gen_amp_mod_At(T, D_roi):

    At = np.zeros((T, D_roi, D_roi))
    At = np.array([0.4*np.eye(D_roi) for _ in range(T)])

    f01 = np.sin(np.linspace(0., 2*np.pi, num=T))
    f10 = -4.*np.sin(np.linspace(0., 2*np.pi, num=T) + 1.2)*f01

    At[:, 0, 1] = f01*np.random.rand()*np.sign(np.random.randn())
    At[:, 1, 0] = f10*np.random.rand()*np.sign(np.random.randn())
    At[-1] = np.zeros((D_roi, D_roi))

    return At


def lds_logZ(Y, A, C, Q, R, mu0, Q0):
    """ Log-partition function computed via Kalman filter that broadcasts over
        the first dimension.
        
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
    p = C.shape[0]

    mu_predict = np.stack([mu0 for _ in range(N)], axis=0)
    sigma_predict = np.stack([Q0 for _ in range(N)], axis=0)

    mus_filt = np.zeros((N, D))
    sigmas_filt = np.zeros((N, D, D))

    ll = 0.

    for t in range(T):

        # condition
        #sigma_pred = dot3(C, sigma_predict, C.T) + R
        tmp1 = einsum2('ik,nkj->nij', C, sigma_predict)
        sigma_y = einsum2('nik,jk->nij', tmp1, C) + R
        sigma_y = sym(sigma_y)

        L = np.linalg.cholesky(sigma_y)
        # res[n] = Y[n,t,:] = np.dot(C, mu_predict[n])
        # the transpose works b/c of how dot broadcasts
        res = Y[...,t,:] - einsum2('ik,nk->ni', C, mu_predict)
        v = solve_triangular(L, res, lower=True)
        
        # log-likelihood over all trials
        ll += (-0.5*np.sum(v*v)
               - np.sum(np.log(np.diagonal(L, axis1=-1, axis2=-2)))
               - p/2.*np.log(2.*np.pi))

        #mus_filt = mu_predict + np.dot(tmp1, solve_triangular(L, v, 'T'))
        mus_filt = mu_predict + einsum2('nki,nk->ni', tmp1,
                                        solve_triangular(L, v, trans='T', lower=True))

        tmp2 = solve_triangular(L, tmp1, lower=True)
        #sigmas_filt = sigma_predict - np.dot(tmp2, tmp2.T)
        sigmas_filt = sigma_predict - einsum2('nki,nkj->nij', tmp2, tmp2)
        sigmas_filt = sym(sigmas_filt)

        # prediction
        #mu_predict = np.dot(A[t], mus_filt[t])
        mu_predict = einsum2('ik,nk->ni', A[t], mus_filt)

        #sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]
        sigma_predict = einsum2('ik,nkl->nil', A[t], sigmas_filt)
        #sigma_predict = einsum2('nil,jl->nij', sigma_predict, A[t]) + Q[t]
        sigma_predict = einsum2('nil,jl->nij', sigma_predict, A[t]) + Q
        sigma_predict = sym(sigma_predict)

    return ll


if __name__ == "__main__":

    #np.random.seed(8675309)
    np.random.seed(42)

    from sim import lds_simulate_loop, rand_stable

    T = 165
    ntrials = 100
    #theta = 1.2
    #A = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    # the same for convenience. constructing reasonable C from small to large
    # dimensions is tricky

    d = 7
    D = 10

    #d = 5
    #D = 6

    A = rand_stable(d)
    A = np.stack([A for _ in range(T)], axis=0)

    #C = np.eye(D)[:,:d]
    C = np.random.randn(D, d)
    C, _ = np.linalg.qr(C)

    Q0 = 0.2*np.eye(d)
    Q = np.stack([0.1*np.eye(d) for _ in range(T)], axis=0)
    Q_true = Q.copy()

    R = 0.1*np.eye(D)

    mu0 = np.zeros(d)

    x, Y = lds_simulate_loop(T, A, C, Q, R, mu0, Q0, ntrials)

    def logZ(params):

        A, L_Q_full = params

        L_Q = np.stack([L*np.tril(np.ones_like(L)) for L in L_Q_full], axis=0)
        Q = einsum2('nik,njk->nij', L_Q, L_Q)
        
        return lds_logZ(Y, A, C, Q, R, mu0, Q0) / Y.shape[0]

    lam = 1e3
    def penalty(params):
        At, _ = params
        return lam*np.sum((At[1:] - At[:-1])**2)

    #A_init = rand_stable(d)
    #A_init = np.stack([A_init for _ in range(T)], axis=0)
    A_init = np.stack([rand_stable(d) for _ in range(T)], axis=0)
    Q_init = np.stack([rand_psd(d) for _ in range(T)], axis=0)
    L_Q_init = np.linalg.cholesky(Q_init)

    #A_init = A.copy()
    #L_Q_init = np.linalg.cholesky(Q)

    params = (A_init, L_Q_init)

    #objective = lambda params, i: -logZ(params) + penalty(params)
    #def callback(params, i, g):
    #    print("it: {}, Log likelihood (penalized) {}".format(i+1, -objective(params, i)))

    #new_params = sgd(grad(objective), params, step_size=1e-5,
    #                 callback=callback, num_iters=100)

    params_flat, unflatten = flatten(params)
    def objective(params_flat):
        params = unflatten(params_flat)
        return -logZ(params) + penalty(params)

    fig = plt.figure()
    fig_quad, axes_quad = plt.subplots(d, d)
    def make_callback():
        it = 0
        def callback(params):
            nonlocal it
            it += 1
            print("it: {} Log likelihood (penalized) {}".format(it, -objective(params)))

            A_est, L_Q_est = unflatten(params)
            Q_est = np.dot(L_Q_est, L_Q_est.T)
            fig.clear()
            ax_true = fig.add_subplot(3,1,1)
            ax_init = fig.add_subplot(3,1,2)
            ax_est = fig.add_subplot(3,1,3)
            ax_true.plot(np.reshape(A[:-1], (-1, d**2)))
            ax_true.set_title("True $A_t$")
            ax_true.set_ylim(-0.5, 0.5)
            ax_init.plot(np.reshape(A_init[:-1], (-1, d**2)))
            ax_init.set_title("Init. $A_t$")
            ax_init.set_ylim(-0.5, 0.5)
            ax_est.plot(np.reshape(A_est, (-1, d**2)))
            ax_est.set_title("Est. $A_t$")
            ax_est.set_ylim(-0.5, 0.5)

            for i in range(d):
                for j in range(d):
                    axes_quad[i,j].cla()
                    axes_quad[i,j].plot(A[:-1, i, j])
                    axes_quad[i,j].plot(A_est[:-1, i,j])
                    axes_quad[i,j].set_title("A[%d,%d]" % (i, j))
                    axes_quad[i,j].set_ylim(-0.5, 0.5)

            plt.ion()
            fig.canvas.draw()
            fig_quad.canvas.draw()
            plt.pause(1./60.)

        return callback
    callback = make_callback()

    spoptions = dict(maxiter=100)
    #spoptions = dict(ftol=1e-6, gtol=1e-6)
    #spoptions = dict(ftol=1e-4, gtol=1e-4)
    #spoptions = dict(ftol=1e-3, gtol=1e-3)
    res = minimize(objective, params_flat, method='L-BFGS-B',
    #spoptions = dict(inexact=False)
    #res = minimize(objective, params_flat, method='trust-krylov',
                   jac=grad(objective),
                   hessp=lambda x, v: make_hvp(objective)(x)[0](v),
                   callback=callback,
                   #tol=1e-6,
                   options=spoptions)
    new_params = unflatten(res.x)
    print("opt success: ", res.success)
    print("opt message: ", res.message)
    #A_est_full, Q_est = new_params
    A_est_full, L_Q_est = new_params
    Q_est = np.dot(L_Q_est, L_Q_est.T)

    # try stochastic gradient (sample minibatch, scale objective and gradient,
    # no line-search), then fine-tune with a quasi-second order method for a
    # few iters

    #grad_AQ = grad(objective)
    #tau = 0.8
    #for gd_iter in range(100):
    #    step_size = 1.
    #    obj_start = objective(params_flat)
    #    grad_flat = grad_AQ(params_flat)
    #    tmp_diff = np.inf
    #    while tmp_diff > 0:
    #        params_new_flat = params_flat - step_size * grad_flat
    #        ref = obj_start - step_size/2. * np.sum(params_new_flat**2)

    #        obj = objective(params_new_flat)
    #        tmp_diff = obj - ref
    #        step_size *= tau
    #    print('iter:', gd_iter, 'step size:', step_size/0.8, ' -- gradA_norm:', np.linalg.norm(grad_flat), ' -- obj:', obj)
    #    callback(params_new_flat)
    #    params_flat = params_new_flat

    ##A_est_full, Q_est = unflatten(params_flat)
    #A_est_full, L_Q_est = unflatten(params_flat)
    #Q_est = np.dot(L_Q_est, L_Q_est.T)

    #A_est = new_params
    #A_est, L_Q_est = new_params

    A_est = A_est_full[:-1]
    #Q_est = einsum2('nik,njk->nij', L_Q_est, L_Q_est)

    #fig = plt.figure()
    fig.clear()
    ax_true = fig.add_subplot(3,1,1)
    ax_init = fig.add_subplot(3,1,2)
    ax_est = fig.add_subplot(3,1,3)
    ax_true.plot(np.reshape(A[:-1], (-1, d**2)))
    ax_true.set_title("True $A_t$")
    ax_true.set_ylim(-0.5, 0.5)
    ax_init.plot(np.reshape(A_init[:-1], (-1, d**2)))
    ax_init.set_title("Init. $A_t$")
    ax_init.set_ylim(-0.5, 0.5)
    ax_est.plot(np.reshape(A_est, (-1, d**2)))
    ax_est.set_title("Est. $A_t$")
    ax_est.set_ylim(-0.5, 0.5)


    from lds import rts_smooth
    Q_est = np.dot(L_Q_est, L_Q_est.T)
    Q_t = np.stack([Q_est for _ in range(T)], axis=0)
    _, x_smooth, _, _ = rts_smooth(Y, A_est_full, C, Q_t, R, mu0, Q0)
    _, x_smooth_true, _, _ = rts_smooth(Y, A, C, np.stack([Q for _ in range(T)], axis=0), R, mu0, Q0)
    fig_sm, axes_sm = plt.subplots(2,1, sharey=True)
    for j in range(d):
        axes_sm[j].plot(np.mean(x[:,:,j], axis=0))
        axes_sm[j].plot(np.mean(x_smooth[:,:,j], axis=0))
        axes_sm[j].plot(np.mean(x_smooth_true[:,:,j], axis=0))

    plt.ion()
    plt.show()

