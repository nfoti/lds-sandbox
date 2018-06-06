import autograd.numpy as np
from autograd import grad

from autograd_linalg import solve_triangular
from lds import rts_smooth
from matplotlib import pyplot as plt
from lds import em_objective

try:
    from einsum2 import einsum2
except ImportError:
    # rename standard numpy function if don't have einsum2
    print("=> WARNING: using standard numpy.einsum,",
          "consider installing einsum2 package")
    from numpy import einsum as einsum2

'''
ToDo: einsum2 everything for #gains
'''

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
    T = Y.shape[1]
    D = A.shape[1]

    predict_mu = np.zeros((N, T + 1, D))
    predict_sigma = np.zeros((N, T + 1, D, D))

    measure_mu = np.zeros((N, T, D))
    measure_sigma = np.zeros((N, T, D, D))

    ll = 0
    for n in range(N):
        predict_mu[n, 0] = mu0
        predict_sigma[n, 0] = Q0
        for t in range(T):
            # 1 MEASUREMENT STEP
            # 1.1 Gain Matrix
            # 1.1.1 Calculate S_t (eq 18.36 in Murphy)
            S_t = np.dot(C, predict_sigma[n, t])
            S_t = np.dot(S_t, C.T) + R
            S_t = sym(S_t) # correct for rounding errors

            # 1.2.3 Calculate gain matrix
            # main idea: want (sigma) (C)' (S_t)^-1
            # this is ((S_t)^-1 (C) (sigma))' 
            # note S_t and sigma are covariance matricies! For S_t see (eq 18.36)

            # we will solve for (S_t)^-1 (C) (sigma) and take the transpose
            # this let's us do the cool cholesky trick which is #faster
            
            # solve the system S_t x = (C)(sigma)
            C_sigma = np.dot(C, predict_sigma[n, t])

            # solve the system L L' x = (C)(sigma), S_t = L L'
            L = np.linalg.cholesky(S_t)

            # first solve L v = C sigma
            v = solve_triangular(L, C_sigma, lower=True)

            # then find x = (S_t)^-1 (C) (sigma_predict)!
            gain_matrix = T_(solve_triangular(L, v, trans='T', lower=True))

            # 1.2 Calculate residuals
            residual = Y[n, t] - np.dot(C, predict_mu[n, t])

            # 1.3 Update mu
            measure_mu[n, t] = predict_mu[n, t] + np.dot(gain_matrix, residual) 

            # 1.4 update sigma
            tmp = np.identity(D) - np.dot(gain_matrix, C)
            measure_sigma[n, t] = np.dot(tmp, predict_sigma[n, t])
            measure_sigma[n, t] = sym(measure_sigma[n, t])

            # 2 PREDICT STEP
            predict_mu[n, t + 1] = np.dot(A[t], measure_mu[n, t])
            predict_sigma[n, t + 1] = np.dot(np.dot(A[t], measure_sigma[n, t]), A[t].T) + Q[t]
            predict_sigma[n, t + 1] = sym(predict_sigma[n, t + 1])

            sv = solve_triangular(L, residual, lower=True)
            # log-likelihood over all trials
            ll += -0.5*np.dot(sv, sv) - 2.*np.sum(np.log(np.diag(L))) \
                  - 0.5*np.log(2.*np.pi)

    return ll, predict_mu, predict_sigma, measure_mu, measure_sigma


def rts_smooth_em(Y, A, C, Q, R, mu0, Q0):
    """ Kalman filter that broadcasts over the first dimension.
        
        Note: This function doesn't handle control inputs (yet).
        
        Y : ndarray, shape=(N, T, D)
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
    N = Y.shape[0]
    T = Y.shape[1]
    D = A.shape[1]

    ll, predict_mu, predict_sigma, measure_mu, measure_sigma = kalman_filter_basic(Y, A, C, Q, R, mu0, Q0)

    smooth_mu = np.zeros((N, T, D))
    smooth_sigma = np.zeros((N, T, D, D))
    backward_gain_matrix = np.zeros((N, T, D, D))

    for n in range(N):
        smooth_mu[n, T - 1] = measure_mu[n, T - 1]
        smooth_sigma[n, T - 1] = measure_sigma[n, T - 1]
        for t in range(T - 2, -1, -1): 
            # 1 CALCULATE THE BACKWARD GAIN MATRIX
            # want (sigma_filt[t]) (A[t + 1])' (sigma_pred[t + 1])^-1
            # calculate the transpose (sigma_pred[t + 1])^-1 (A[t + 1])(sigma_filt[t])
            # using the cholesky trick                  
            A_sigma = np.dot(A[t], measure_sigma[n, t]) # why is this A_t and not A_{t + 1}
            
            # solve the system (sigma_pred[t + 1]) x = A_sigma
            L = np.linalg.cholesky(predict_sigma[n, t + 1])
            v = solve_triangular(L, A_sigma, lower=True)
            gain_matrix = (solve_triangular(L, v, trans='T', lower=True)).T

            backward_gain_matrix[n, t] = gain_matrix

            # update the smooth'd values
            smooth_mu[n, t] = measure_mu[n, t] + np.dot(gain_matrix, smooth_mu[n, t + 1] - predict_mu[n, t + 1])
            smooth_sigma[n, t] = np.dot(gain_matrix, smooth_sigma[n, t + 1] - predict_sigma[n, t + 1])
            smooth_sigma[n, t] = measure_sigma[n, t] + np.dot(smooth_sigma[n, t], gain_matrix.T)

    return backward_gain_matrix, ll, predict_mu, predict_sigma, measure_mu, measure_sigma, smooth_mu, smooth_sigma


def rts_smooth_basic(Y, A, C, Q, R, mu0, Q0):
    return rts_smooth_em(Y, A, C, Q, R, mu0, Q0)[1:]


def em_stationary(Y, A, C, Q, R, mu0, Q0):
    """
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

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """
    N = Y.shape[0]
    T = Y.shape[1]
    D = A.shape[1]

    cur_A = A
    cur_Q = Q
    cur_m0 = mu0
    cur_Q0 = Q0

    for i in range(50):
        # the e-step for all time steps
        params = rts_smooth_em(Y, cur_A, C, cur_Q, R, cur_m0, cur_Q0)
        gain_matrix, ll, _, _, _, _, smooth_mu, smooth_sigma = params
        ll, smooth_mu, smooth_sigma, sigmas_smooth_tnt = rts_smooth

        # first calculate the beginning stuff
        cur_m0 = np.mean(smooth_mu[:, 0], axis = 0)
        cur_Q0 = np.mean(smooth_sigma[:, 0], axis = 0) + np.dot(cur_m0, cur_m0.T)     

        # M step: use bishop's reference eq 13.110 and on
        for t in range(1, T):
            # look at eq 13.113 in bishop. Look at eq 13.104 - 13.107 for how to compute these efficiently
            # these are used in the update of A
            state_two_slice = np.zeros((D, D)) # sum of E[z_n z_{n-1}']   
            state_state_squared = np.zeros((D, D)) # sum of E[z_{n-1} z_{n-1}']  # this is psd!   

            # to do, this is just state_state_squared - one time step + one time step   
            state_state_squared_ahead = np.zeros((D, D)) # sum of E[z_{n} z_{n}']  # this is psd!                 

            for n in range(N):
                state_two_slice += np.dot(gain_matrix[n, t- 1], smooth_sigma[n, t]) \
                                    + np.dot(smooth_mu[n, t], smooth_mu[n, t - 1].T)
                state_state_squared += smooth_sigma[n, t - 1] + np.dot(smooth_mu[n, t - 1], smooth_mu[n, t - 1].T)
                state_state_squared_ahead += smooth_sigma[n, t] + np.dot(smooth_mu[n, t], smooth_mu[n, t].T)

        # update parameters. Note that state_state_squared is psd and symmetric!
        # A_new = (state_two_slice)(state_state_squared)^-1
        # A_new' = (state_state_squared')^-1 (state_two_slice)' --> x' = B^-1 C' (recall B = symmetric)
        # solve the system B x' = C' using cholesky decomposition on B'
        # L' (L x') = C' --> L' v = C'
        L = np.linalg.cholesky(state_state_squared)
        v = solve_triangular(L, state_two_slice.T, lower=True)
        cur_A[:, ...] = (solve_triangular(L, v, trans='T', lower=True)).T

        # the update for Q is long and torturous :( see 13.114 of bishop for the monstrosity
        # uncommenting this leads to explosions eventually
        # TODO: fix this
        #updateQ = state_state_squared_ahead - np.dot(cur_A[0], state_two_slice.T)
        #updateQ -= np.dot(state_two_slice, cur_A[0])
        #updateQ += np.dot(np.dot(cur_A[0], state_state_squared), cur_A[0].T)
        #updateQ /= (N - 1)
        #cur_Q[:, ...] = sym(updateQ)

        print(cur_A[0])
        print(ll)

def em_stochasic_temporal_closed_form(Y, A, C, Q, R, mu0, Q0, lambda_temporal = 0, lambda_l2 = 0, iterations_em = 50, iterations_gd = 50, learning_rate = 1, gd_threshold = 0.001, A_true = None, X_true = None, Q_true = None):
    '''
    Note: this does not work
    '''

    '''
    Run smoothing on all trials with initial parameters to obtain estimates on state
    C and R are fixed!

    Only does A for now
    '''
    N = Y.shape[0]
    T = Y.shape[1]
    D = A.shape[1]

    cur_A = A
    cur_Q = Q
    cur_m0 = mu0
    cur_Q0 = Q0
    fig_quad, axes_quad = plt.subplots(D, D, figsize=(12,6))

    for em_it in range(iterations_em):
        for i in range(D):
            for j in range(D):
                axes_quad[i, j].cla()

                if A_true is not None:
                    axes_quad[i, j].plot(A_true[:, i, j], color='green')
                axes_quad[i, j].plot(cur_A[:, i, j], color='red')
                axes_quad[i, j].set_ylim(-1.5, 1.5)
        fig_quad.canvas.draw()
        if em_it in [0, 1, 2, 3, 5, 10]:
            print(em_it)
            fig_quad.savefig('iteration' + str(em_it) + '.png')
        plt.pause(1)

        # the e-step for all time steps
        #params = rts_smooth_em(Y, cur_A, C, cur_Q, R, cur_m0, cur_Q0)
        #gain_matrix, ll, _, _, _, _, smooth_mu, smooth_sigma = params
        ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = rts_smooth(Y, cur_A, C, cur_Q, R, cur_m0, cur_Q0)

        if X_true is not None:
            mus_smooth = X_true

        # first calculate the beginning stuff
        cur_m0 = np.mean(mus_smooth[:, 0], axis = 0)
        cur_Q0 = np.mean(sigmas_smooth[:, 0], axis = 0) + np.dot(cur_m0, cur_m0.T)

        # M step: use gradient descent
        new_A = cur_A.copy()

        # currently update every A_t's
        for t in range(T - 2, -1, -1):
            gd_i = 0
            while gd_i < iterations_gd:
                gd_i += 1

                # updated A_t for this iteration of em
                A_t = new_A[t].copy()

                # average up the gradient as calculated from all of the trials
                gradient = A_t.copy()
                gradient.fill(0)
                sigma_inverses = A_t.copy()
                sigma_inverses.fill(0)
                for n in range(N):
                    mu_t_prev = np.reshape(mus_smooth[n, t], (D, 1))
                    mu_t_next = np.reshape(mus_smooth[n, t + 1], (D, 1))

                    sigma_t = sigmas_smooth[n, t]

                    if Q_true is not None:
                        sigma_t = Q_true[t]

                    sigma_inverse = np.linalg.inv(sigma_t)
                    sigma_inverses += sigma_inverse

                    # TODO: cholesky this
                    error = mu_t_next - np.dot(A_t, mu_t_prev)

                    gradient_n = np.dot(mu_t_prev, error.T)

                    #numerical concerns with inverting a very small sigma
                    gradient_n = -0.5 * np.dot(gradient_n, sigma_inverse)

                    gradient += gradient_n

                gradient /= N
                sigma_inverses /= N
                new_A[t] -= (0.8 ** gd_i) * (1 / np.sum(sigma_inverses)) * learning_rate * gradient
        
        cur_A = new_A


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


def em_stochastic_temporal(Y, A, C, Q, R, mu0, Q0,
                lambda_temporal=0, lambda_l2=0, iterations_em=20, iterations_gd=5, learning_rate=1, gd_threshold=0.001,
                A_true=None, X_true=None, Q_true=None, plot_progress=False):
    """ Stochastic EM with time varying A and Q

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D, D)
          Initial guess of time-varying dynamic matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        Q : ndarray, shape=(T, D, D)
          Initial guess of time-varying covariances of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        lambda_temporal: float
        lambda_l2: float

        if A_t is not None, will plot true dynamics under plot_progress
        if X_true is not None, will use X_true instead of smoothed X's for debugging
        if Q_true is not None, will use Q_true instead of smoothed X's for debugging

        plot_progress plots true (if given) and currently predicted A and Q
    """

    N = Y.shape[0]
    T = Y.shape[1]
    D = A.shape[1]

    cur_A = A
    cur_Q = Q
    cur_m0 = mu0
    cur_Q0 = Q0
    fig_quad, axes_quad = None, None
    if plot_progress:
        fig_quad, axes_quad = plt.subplots(D, D, figsize=(12, 6))

    for em_it in range(iterations_em):
        # the e-step for all time steps
        # params = rts_smooth_em(Y, cur_A, C, cur_Q, R, cur_m0, cur_Q0)
        # gain_matrix, ll, _, _, _, _, smooth_mu, smooth_sigma = params
        ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = rts_smooth(Y, cur_A, C, cur_Q, R, cur_m0, cur_Q0)
        print('Iteration %d with ll %f' % (em_it, ll))

        if plot_progress:
            lds_plot_progress(A_true, cur_A, Q_true, cur_Q, D, fig_quad, axes_quad,
                                save=em_it in [0, 1, 2, 3, 5, 10], save_name='iteration' + str(em_it))

        if X_true is not None:
            print('True X:', X_true[0])
            mus_smooth = X_true
            print('Smoothed X:', mus_smooth[0])

        # M step: use gradient descent on -ll
        new_A = cur_A.copy()
        new_Q = cur_Q.copy()

        # currently update every A_t's
        # TODO: unroll loops with einsum2
        for t in range(T - 2, -1, -1):
            gd_i = 0
            while gd_i < iterations_gd:
                gd_i += 1

                # updated A_t for this iteration of em
                A_t = new_A[t].copy()

                # average up the gradient as calculated from all of the trials
                gradient = A_t.copy()
                gradient.fill(0)
                sigma_inverses = A_t.copy()
                sigma_inverses.fill(0)
                for n in range(N):
                    mu_t_prev = np.reshape(mus_smooth[n, t], (D, 1))
                    mu_t_next = np.reshape(mus_smooth[n, t + 1], (D, 1))

                    sigma_t = sigmas_smooth[n, t]

                    def obj_func_At(A_t, sigma_t):
                        # this returns the -ln(N(z_t | A_t, Z_{t-1}, Q_t))
                        # we want to minimize this!
                        error = mu_t_next - np.dot(A_t, mu_t_prev)
                        obj = np.dot(error.T, np.linalg.inv(sigma_t))
                        obj = np.dot(obj, error)
                        return obj

                    grad_fun = grad(lambda A_t: obj_func_At(A_t, sigma_t))

                    gradient += grad_fun(A_t)

                gradient /= N

                # TODO: verify this and update Q, from lds.py
                sigma_inverses /= N
                new_A[t] -= (0.8 ** (gd_i)) * learning_rate * gradient

                # for calculating updates to Q_t, taken from lds
                w_s = 1.
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

                # use A_t to get Q_t, this is taken from lds sandbox!
                AtB2T = einsum2('tik,tjk->tij', new_A[:-1], B2)
                B2AtT = einsum2('tik,tjk->tij', B2, new_A[:-1])

                tmp = einsum2('tik,tkl->til', new_A[:-1], B3)
                AtB3AtT = einsum2('til,tjl->tij', tmp, new_A[:-1])
                elbo_2 = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)
                new_Q[t] = 1. / (N * T) * elbo_2

        # update current A matrix for new E step
        cur_A[:] = new_A[:]
        print("GRADIENT MOI", new_A[0])

        # update the sigma matrix from closed form
        #cur_Q[:] = new_Q[:]

        # first calculate the beginning stuff
        cur_m0 = np.mean(mus_smooth[:, 0], axis=0)
        cur_Q0 = np.mean(sigmas_smooth[:, 0], axis=0) + np.dot(cur_m0, cur_m0.T)

