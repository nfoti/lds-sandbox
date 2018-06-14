import timeit
import autograd.numpy as np
from lds import rand_psd, kalman_filter, rts_smooth, lds_simulate_loop, em
from stochastic_em import kalman_filter_basic, rts_smooth_basic, em_stochastic_temporal
from lds_stochastic_em import em_stochastic

# TODO: remove this when done!!!!!!
np.random.seed(325199)


def test_kalman_filter():
    N = 10
    T = 500
    D = 10

    Y = np.random.rand(N, T, D)
    A = np.random.rand(T, D, D)
    C = np.random.rand(D, D)
    mu0 = np.random.rand(D)
    Q0 = rand_psd(D)
    Q = np.array([rand_psd(D) for i in range(T)])
    R = rand_psd(D)

    print('KALMAN FILTERING')
    t1 = timeit.default_timer()
    ll, mus_filt, sigmas_filt = kalman_filter(Y, A, C, Q, R, mu0, Q0)
    t2 = timeit.default_timer()
    print('\tlds took', t2 - t1)

    t1 = timeit.default_timer()
    ll_me, predict_mu, predict_sigma, measure_mu, measure_sigma = kalman_filter_basic(Y, A, C, Q, R, mu0, Q0)
    t2 = timeit.default_timer()
    print('\tme took', t2 - t1)

    print('\tabs error mu', np.sum(np.abs(measure_mu - mus_filt)))
    print('\tabs error sigma', np.sum(np.abs(measure_sigma - sigmas_filt)))
    print('\tabs error ll', np.sum(np.abs(ll - ll_me)))


def test_kalman_smoother():
    N = 10
    T = 10
    D = 10
    p = 5    

    Y = np.random.rand(N, T, D)
    A = np.random.rand(T, D, D)
    C = np.random.rand(D, D)
    mu0 = np.random.rand(D)
    Q0 = rand_psd(D)
    Q = np.array([rand_psd(D) for i in range(T)])
    R = rand_psd(D)

    print('KALMAN SMOOTHING')
    t1 = timeit.default_timer()
    ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt = rts_smooth(Y, A, C, Q, R, mu0, Q0)
    t2 = timeit.default_timer()
    print('\tlds took', t2 - t1, 's')

    t1 = timeit.default_timer()
    ans = rts_smooth_basic(Y, A, C, Q, R, mu0, Q0)
    ll_me, predict_mu, predict_sigma, measure_mu, measure_sigma, smooth_mu, smooth_sigma = ans
    t2 = timeit.default_timer()
    print('\tme took', t2 - t1, 's')

    print('\tabs error mu', np.sum(np.abs(smooth_mu - mus_smooth)))
    print('\tabs error sigma', np.sum(np.abs(smooth_sigma - sigmas_smooth)))


if __name__ == '__main__':
    #test_kalman_filter()
    test_kalman_smoother()

    N = 2000
    T = 100
    D = 20

    A_stationary = np.identity(D) #np.random.rand(D, D)

    A_true = [A_stationary.copy() for i in range(T)]
    A_true = np.array(A_true)

    # uncomment to get random A
    #A_true = np.random.rand(T, D, D) * 0.5

    mu0 = np.random.rand(D)
    C_true = np.identity(D)

    # for now have very low variance!
    Q0 = np.identity(D)
    Q_true = np.array([Q0.copy() for i in range(T)])
    R_true = np.identity(D)

    X, Y = lds_simulate_loop(T, A_true, C_true, Q_true, R_true, mu0, Q0, N)

    print('EM')
    print("TRUE A:\n", A_true[0])
    print()

    A_init = np.random.rand(T, D, D)
    Q_init = np.array([rand_psd(D) for i in range(T)])
    print('Init A:\n', A_init[0])
    print('Mu 0:\n', Q_init[0])

    initparams = (A_init, Q0, Q0)
    fixedparams = (C_true, R_true, mu0)
    ldsregparams = (0.1, 0.1)

    print('**************temporal model**************')
    t1 = timeit.default_timer()
    em_stochastic(Y, initparams, fixedparams, ldsregparams,
                  Atrue=A_true, plot_progress=False, save_plots=False, debug=False, maxiter=10,
                  num_stochastic=10, num_objvals=5, tol=1e-6)
    t2 = timeit.default_timer()
    print('\tStochastic took', t2 - t1)
    print()

    print('**************truth**************')
    t1 = timeit.default_timer()
    em(Y, initparams, fixedparams, ldsregparams, plot_progress=False, save_plots=False, niter=5, Atrue=A_true, num_objvals=5, tol=1e-6)
    t2 = timeit.default_timer()
    print('\tFull took', t2 - t1)
    print()

    #print('**************stationary model**************')
    #em_stationary(Y, A_init, C_true, Q_true, R_true, mu0, Q0, iterations = 50, threshold_stop = 0.001)
    #print()



    
