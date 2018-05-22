import numpy as np
import timeit

from lds import *
from rts_smoother import kalman_filter_basic, rts_smooth_basic, em_stationary

np.random.seed(325199)


def test_kalman_filter():
    N = 10
    T = 500
    D = 10
    p = 5    

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
    T = 500
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
    print('\tlds took', t2 - t1)

    t1 = timeit.default_timer()
    ans = rts_smooth_basic(Y, A, C, Q, R, mu0, Q0)
    ll_me, predict_mu, predict_sigma, measure_mu, measure_sigma, smooth_mu, smooth_sigma = ans
    t2 = timeit.default_timer()
    print('\tme took', t2 - t1)

    print('\tabs error mu', np.sum(np.abs(smooth_mu - mus_smooth)))
    print('\tabs error sigma', np.sum(np.abs(smooth_sigma - sigmas_smooth)))


if __name__ == '__main__':
    #test_kalman_filter()
    #test_kalman_smoother()

    N = 10
    T = 500
    D = 10
    p = 5    

    A = np.random.rand(T, D, D)
    C = np.random.rand(D, D)
    mu0 = np.random.rand(D)
    Q0 = rand_psd(D)
    Q = np.array([rand_psd(D) for i in range(T)])
    R = rand_psd(D)

    print('EM')
    '''
    '''

    
