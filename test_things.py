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
    Q = rand_psd(D)
    R = rand_psd(D)

    ll, mus_filt, sigmas_filt = kalman_filter_loop(Y, A, C, Q, R, mu0, Q0)

    print(0)
    print(np.reshape(mus_filt, (T, D)) [ 0])
    print()
    print(np.reshape(sigmas_filt, (T, D, D))[0])
    print()

    measure_mu, measure_sigma = kalman_filter_basic(np.reshape(Y, (T, D)), A, C, Q, R, mu0, Q0)
