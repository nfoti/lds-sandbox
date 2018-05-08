import numpy as np

from lds import *
from rts_smoother import kalman_filter_basic

np.random.seed(3251997)

if __name__ == '__main__':
    N = 1
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

    ll, mus_filt, sigmas_filt = kalman_filter(Y, A, C, Q, R, mu0, Q0)
    measure_mu, measure_sigma = kalman_filter_basic(Y, A, C, Q, R, mu0, Q0)

    print(np.sum(np.abs(measure_mu - mus_filt)))
    print(np.sum(np.abs(measure_sigma - sigmas_filt)))
