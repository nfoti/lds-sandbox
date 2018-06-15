from __future__ import division
from __future__ import print_function

import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from lds import rts_smooth, lds_plot_progress

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


# performs e steps and returns necessary statistics to calculate gradients and updates
def e_step(Y, A, C, Q, R, mu0, Q0):
    smoothed_state_params = rts_smooth(Y, A, C, Q, R, mu0, Q0)
    ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = smoothed_state_params

    D = Y.shape[len(Y.shape) - 1]
    w_s = 1.
    x_smooth_0_outer = einsum2('ri,rj->rij', mus_smooth[:, 0, :D],
                               mus_smooth[:, 0, :D])
    B0 = w_s * np.sum(sigmas_smooth[:, 0, :D, :D] + x_smooth_0_outer,
                      axis=0)

    x_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:, 1:, :D],
                             mus_smooth[:, 1:, :D])
    B1 = w_s * np.sum(sigmas_smooth[:, 1:, :D, :D] + x_smooth_outer, axis=0)

    z_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:, :-1, :],
                             mus_smooth[:, :-1, :])
    B3 = w_s * np.sum(sigmas_smooth[:, :-1, :, :] + z_smooth_outer, axis=0)

    mus_smooth_outer_l1 = einsum2('rti,rtj->rtij',
                                  mus_smooth[:, 1:, :D],
                                  mus_smooth[:, :-1, :])
    B2 = w_s * np.sum(sigmas_smooth_tnt[:, :, :D, :] + mus_smooth_outer_l1, axis=0)

    B = (B0, B1, B2, B3)
    smooth_params = (ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt)
    return B, smooth_params


# joint objective for At and L_Q
def L2_obj(At, L_Q, B1, B2, B3, lam0, lam1):
    AtB2T = einsum2('tik,tjk->tij', At, B2)
    B2AtT = einsum2('tik,tjk->tij', B2, At)

    tmp = einsum2('tik,tkl->til', At, B3)
    AtB3AtT = einsum2('til,tjl->tij', tmp, At)
    elbo_2 = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)

    L_Q_inv_elbo_2 = solve_triangular(L_Q, elbo_2, lower=True)
    obj = np.trace(solve_triangular(L_Q, L_Q_inv_elbo_2, lower=True, trans='T'))
    obj += lam0 * np.sum(At ** 2)
    AtmAtm1_2 = (At[1:] - At[:-1]) ** 2
    obj += lam1 * np.sum(AtmAtm1_2)
    return obj


def em_stochastic(Y, initparams, fixedparams, ldsregparams, maxiter = 50,
                  Atrue=None, plot_progress=False, save_plots=False, debug=False,
                  mini_batch_size=10, num_objvals=5, tol=1e-6):
    A_init, Q_init, Q0_init = initparams
    A = A_init.copy()
    Q = Q_init.copy()
    Q0 = Q0_init.copy()

    C, R, mu0 = fixedparams
    lam0, lam1 = ldsregparams

    L_Q = np.linalg.cholesky(Q)
    L_R = np.linalg.cholesky(R)

    _, D, Dnlags = A.shape
    ntrials, T, p = Y.shape

    bestparams = (A.copy(), Q.copy(), Q0.copy())
    At = A[:-1]

    fig_quad, axes_quad = None, None
    if plot_progress:
        fig_quad, axes_quad = plt.subplots(D, D, figsize=(12, 6))
        lds_plot_progress(Atrue, At, None, None, D, fig_quad, axes_quad,
                          save=save_plots, save_name='stochasticem_iteration0')

    # variables for loop
    obj_diff = np.finfo('float').max
    obj = 1
    it = 0

    em_obj_list = np.zeros(maxiter)

    # gradient descent with stochastic EM
    while np.abs(obj_diff / obj) > 1e-6 and it < maxiter:
        # e-step
        cur_rows = np.random.choice(ntrials, mini_batch_size, replace=False)
        Y_cur = Y[cur_rows]

        L_Q0 = np.linalg.cholesky(Q0)
        Q = np.dot(L_Q, L_Q.T)

        B, smooth_params = e_step(Y_cur, A, C, Q, R, mu0, Q0)
        B0, B1, B2, B3, = B
        ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = smooth_params

        # track obj. for debugging purposes
        if debug:
            it_params = (At, L_Q, Q0)
            it_fixedparams = (C, L_R)
            em_obj, L1, L2, L3, penalty = em_objective(Y, D, it_params, it_fixedparams, ldsregparams,
                                  mus_smooth, sigmas_smooth, sigmas_smooth_tnt)
            em_obj_list[it] = em_obj

            # check for updated best iterate
            if em_obj < best_em_obj:
                best_em_obj = em_obj
                bestparams = (A.copy(), Q.copy(), Q0.copy())

            # check for convergence
            if it >= num_objvals:
                vals_to_check = em_obj_list[it-num_objvals:it]
                if np.all(np.abs((vals_to_check - em_obj) / em_obj) <= tol):
                    print('EM objective converged')
                    em_obj_list = em_obj_list[:it+1]
                    break

            print('em iter:', it + 1, 'EM objective: ', em_obj)
            print('  L1:', L1)
            print('  L2:', L2)
            print('  L3:', L3)
            print('  pen:', penalty)


        # m-step
        Q0 = 1. / (mini_batch_size) * B0

        # gradient update on A
        # for comparing before and after
        obj_start = L2_obj(At, L_Q, B1, B2, B3, lam0, lam1)

        grad_function = grad(lambda At: L2_obj(At, L_Q, B1, B2, B3, lam0, lam1))
        grad_A_un = grad_function(At)
        grad_A = grad_A_un  # / np.linalg.norm(grad_A_un.flatten())
        tmp_diff = np.inf

        step_size = 10000.
        tau = 0.8
        while tmp_diff > 0:
            new_At = At - step_size * grad_A
            ref = obj_start - step_size / 10000. * (np.sum(grad_A_un * grad_A))
            obj = L2_obj(new_At, L_Q, B1, B2, B3, lam0, lam1)
            tmp_diff = obj - ref
            step_size *= tau

        At[:] = new_At

        # update Q using closed form
        AtB2T = einsum2('tik,tjk->tij', At, B2)
        B2AtT = einsum2('tik,tjk->tij', B2, At)
        tmp = einsum2('tik,tkl->til', At, B3)
        AtB3AtT = einsum2('til,tjl->tij', tmp, At)
        elbo_2 = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)
        Q = 1. / (mini_batch_size * T) * elbo_2
        L_Q = np.linalg.cholesky(Q)

        # update loop variables and plot stuff
        obj = L2_obj(At, L_Q, B1, B2, B3, lam0, lam1)
        obj_diff = obj_start - obj

        if plot_progress:
            lds_plot_progress(Atrue, At, None, None, D, fig_quad, axes_quad,
                              save=plot_progress, save_name='stochasticem_iteration' + str(it + 1))

        it += 1
        if debug:
            print('cur it', it)

    # retrieve best parameters (only works if debug is on)
    A, Q, Q0 = bestparams

    print('final smoothing with estimated parameters')
    _, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = rts_smooth(Y, A, C, Q, R, mu0, Q0)

    ret = {'A': A, 'L_Q': L_Q, 'L_Q0': L_Q0, 'mus_smooth': mus_smooth,
           'sigmas_smooth': sigmas_smooth, 'sigmas_smooth_tnt': sigmas_smooth_tnt,
           'em_obj_vals': em_obj_list}

    return ret

if __name__ == "__main__":
    np.random.seed(42)

    T = 165
    ntrials = 20
    d = 10
    D = 120

    A = np.array([0.5*np.eye(d) for _ in range(T)])
    f01 = np.sin(np.linspace(0., 2*np.pi, num=T))
    f10 = -np.sin(np.linspace(0., 2*np.pi, num=T) + 1.2)*f01
    A[:,0,1] = f01*np.random.randn()*np.sign(np.random.randn())
    A[:,1,0] = f10*np.random.randn()*np.sign(np.random.randn())
    A[-1] = np.zeros((d, d))
    C, _ = np.linalg.qr(np.random.randn(D, d))
    Q0 = rand_psd(d, maxew=0.5)
    Q = rand_psd(d, maxew=0.5)
    Q_stack = np.stack([Q for _ in range(T)], axis=0)

    R = rand_psd(D)

    mu0 = np.zeros(d)

    x, Y = lds_simulate_loop(T, A, C, Q_stack, R, mu0, Q0, ntrials)
    A_init = np.array([rand_stable(d, s=0.7) for _ in range(T)])  # np.random.randn(*A.shape)
    Q_init = rand_psd(d)
    Q0_init = rand_psd(d)

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

    ret = em_stochastic(Y, initparams, fixedparams, ldsregparams,
                        Atrue=A, plot_progress=True, save_plots=True, debug=False, maxiter=50,
                        mini_batch_size=3, num_objvals=5, tol=1e-6)
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
    plt.savefig('ending1')

    plt.figure()
    plt.plot(em_obj_vals)
    plt.savefig('ending2')
