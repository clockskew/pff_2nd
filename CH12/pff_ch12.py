import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl
import scipy.stats as scs


def print_statistics(a1, a2):
    ''' Prints selected statistics.
    Parameters
    ==========
    a1, a2: ndarray objects
    results objects from simulation
     '''
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print('%14s %14s %14s' % ('statistic', 'data set 1', 'data set 2'))
    print(45 * "-")
    print('%14s %14.3f %14.3f' % ('size', sta1[0], sta2[0]))
    print('%14s %14.3f %14.3f' % ('min', sta1[1][0], sta2[1][0]))
    print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1]))
    print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2]))
    print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4]))
    print('%14s %14.3f %14.3f' % ('kurtosis', sta1[5], sta2[5]))


def ramd():
    sample_size = 500
    rn1 = npr.rand(sample_size, 3)
    rn2 = npr.randint(0, 10, sample_size)
    rn3 = npr.sample(size=sample_size)

    a = [0, 25, 50, 75, 100]

    rn4 = npr.choice(a, size=sample_size)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    ax1.hist(rn1, bins=25, stacked=True)
    ax1.set_title('rand')
    ax1.set_ylabel('frequency')

    ax2.hist(rn2, bins=25)
    ax2.set_title('randint')

    ax3.hist(rn3, bins=25)
    ax3.set_title('samople')
    ax3.set_ylabel("frequency")

    ax4.hist(rn4, bins=25)
    ax4.set_title('choice')
    plt.show()


def distribution():
    sample_size = 500
    rn1 = npr.standard_normal(sample_size)
    rn2 = npr.normal(100, 20, sample_size)
    rn3 = npr.chisquare(df=0.5, size=sample_size)
    rn4 = npr.poisson(lam=1.0, size=sample_size)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    ax1.hist(rn1, bins=25, stacked=True)
    ax1.set_title('Standard normal')
    ax1.set_ylabel('frequency')

    ax2.hist(rn2, bins=25)
    ax2.set_title('Normal 100,20')

    ax3.hist(rn3, bins=25)
    ax3.set_title('Chi squared')
    ax3.set_ylabel("frequency")

    ax4.hist(rn4, bins=25)
    ax4.set_title('Poisson')
    plt.show()


def geometric_brownian_motion():
    S0 = 100
    r = 0.05
    I = 10000
    M = 50
    sigma = .25
    T = 2.0
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * npr.standard_normal(I))
    plt.figure(figsize=(10, 6))
    plt.hist(S[-1], bins=50)
    plt.xlabel("Index level (Brownian)")
    plt.ylabel('frequency')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(S[:, :1000], lw=1.5)
    plt.xlabel('time')
    plt.ylabel('index level')
    plt.show()
    return S


def montacarlo_simulation():
    S0 = 100
    r = 0.05
    sigma = .25
    T = 2.0
    I = 10000
    STD = npr.standard_normal(I)
    ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * STD)

    plt.figure(figsize=(10, 6))
    plt.hist(ST1, bins=50)
    plt.xlabel("Index level")
    plt.ylabel('frequency')
    plt.show()

    plt.figure(figsize=(10, 6))

    return ST1


def stochastic_volatility():
    S0 = 100.
    r = 0.05
    v0 = 0.1
    kappa = 3.0
    theta = 0.25
    sigma = 0.1
    rho = 0.6
    T = 1.0

    corr_mat = np.zeros((2, 2))
    corr_mat[0, :] = [1.0, rho]
    corr_mat[1, :] = [rho, 1.0]
    cho_mat = np.linalg.cholesky(corr_mat)

    M = 50
    I = 10000
    dt = T / M

    ran_num = npr.standard_normal((2, M + 1, I))
    v = np.zeros_like(ran_num[0])
    vh = np.zeros_like(v)
    v[0] = v0
    vh[0] = v0

    for t in range(1, M + 1):
        ran = np.dot(cho_mat, ran_num[:, t, :])
        vh[t] = (vh[t - 1] +
                 kappa * (theta - np.maximum(vh[t - 1], 0)) * dt +
                 sigma * np.sqrt(np.maximum(vh[t - 1], 0)) *
                 math.sqrt(dt) * ran[1])

    v = np.maximum(vh, 0)
    S = np.zeros_like(ran_num[0])
    S[0] = S0
    for t in range(1, M + 1):
        ran = np.dot(cho_mat, ran_num[:, t, :])
        S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt + np.sqrt(v[t]) * ran[0] * np.sqrt(dt))

    ig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.hist(S[-1], bins=50)
    ax1.set_xlabel('index level')
    ax1.set_ylabel('frequency')
    ax2.hist(v[-1], bins=50)
    ax2.set_xlabel('volatility');
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax1.plot(S[:, :100], lw=1)
    ax1.set_ylabel('index level')
    ax2.plot(v[:, :100], lw=1)
    ax2.set_xlabel('time')
    ax2.set_ylabel('volatility')
    plt.show()


def square_root_diffusion_euler():
    x0 = 0.25
    kappa = 3.0
    theta = 0.15
    sigma = 0.1
    I = 10000
    M = 50
    dt = T / M
    xh = np.zeros((M + 1, I))
    x = np.zeros_like(xh)
    xh[0] = x0
    x[0] = x0
    for t in range(1, M + 1):                 xh[t] = (
            xh[t - 1] + kappa * (theta - np.maximum(xh[t - 1], 0)) * dt + sigma * np.sqrt(
        np.maximum(xh[t - 1], 0)) * math.sqrt(dt) * npr.standard_normal(I))
    x = np.maximum(xh, 0)
    plt.figure(figsize=(10, 6))
    plt.hist(x[-1], bins=50)
    plt.xlabel('value(SRT(T)')
    plt.ylabel('frequency')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(x[:, :100], lw=1.5)
    plt.xlabel('time')
    plt.ylabel('index level')
    plt.show()
    return x


def jump_diffusion():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    lamb = 0.05
    mu = -0.6
    delta = 0.25
    rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1)
    T = 1.0
    M = 50
    I = 10000
    dt = T / M

    S = np.zeros((M + 1, I))
    S[0] = S0
    sn1 = npr.standard_normal((M + 1, I))
    sn2 = npr.standard_normal((M + 1, I))
    poi = npr.poisson(lamb * dt, (M + 1, I))
    for t in range(1, M + 1, 1):
        S[t] = S[t - 1] * (
                np.exp((r - rj - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * sn1[t]) + (
                np.exp(mu + delta * sn2[t]) - 1) * poi[t])
    S[t] = np.maximum(S[t], 0)
    plt.figure(figsize=(10, 6))
    plt.hist(S[-1], bins=50)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(S[:, :100], lw=1.)
    plt.xlabel('time')
    plt.ylabel('index level')
    plt.show()


def gen_sn(M, I, anti_paths=True, mo_match=True):
    ''' Function to generate random numbers for simulation.

    Parameters
    ==========
    M: int
        number of time intervals for discretization
    I: int
        number of paths to be simulated
    anti_paths: boolean
        use of antithetic variates
    mo_math: boolean
        use of moment matching
    '''
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2)))
        sn = np.concatenate((sn, -sn), axis=1)
    else:
        sn = npr.standard_normal((M + 1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn


def gbm_mcs_stat(K):
    ''' Valuation of European call option in Black-Scholes-Merton
    by Monte Carlo simulation (of index level at maturity)

    Parameters
    ==========
    K: float
        (positive) strike price of the option

    Returns
    =======
    C0: float
        estimated present value of European call option
    '''
    sn = gen_sn(1, I)
    # simulate index level at maturity
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T
                     + sigma * math.sqrt(T) * sn[1])
    # calculate payoff at maturity
    hT = np.maximum(ST - K, 0)
    # calculate MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0


def gbm_mcs_dyna(K, option='call'):
    ''' Valuation of European options in Black-Scholes-Merton
    by Monte Carlo simulation (of index level paths)

    Parameters
    ==========
    K: float
        (positive) strike price of the option
    option : string
        type of the option to be valued ('call', 'put')

    Returns
    =======
    C0: float
        estimated present value of European call option
    '''
    dt = T / M
    # simulation of index level paths
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                                 + sigma * math.sqrt(dt) * sn[t])
    # case-based calculation of payoff
    if option == 'call':
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    # calculation of MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0


def create_plot(x, y, styles, labels, axlabels):
    plt.figure(figsize=(10, 6))

    plt.scatter(x[0], y[0])
    plt.scatter(x[1], y[1])
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    plt.legend(loc=0)
    plt.show()


def gbm_mcs_amer(K, option='call'):
    ''' Valuation of American option in Black-Scholes-Merton
    by Monte Carlo simulation by LSM algorithm

    Parameters
    ==========
    K : float
        (positive) strike price of the option
    option : string
        type of the option to be valued ('call', 'put')

    Returns
    =======
    C0 : float
        estimated present value of European call option
    '''
    dt = T / M
    df = math.exp(-r * dt)
    # simulation of index levels
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                                 + sigma * math.sqrt(dt) * sn[t])
    # case based calculation of payoff
    if option == 'call':
        h = np.maximum(S - K, 0)
    else:
        h = np.maximum(K - S, 0)
    # LSM algorithm
    V = np.copy(h)
    for t in range(M - 1, 0, -1):
        res = np.polyfit(S[t], V[t + 1] * df, 7)
        ry = np.polyval(res, S[t])
        create_plot([S[t], S[t],S[t]], [V[t + 1] * df, ry,], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])
        V[t] = np.where(ry > h[t], V[t + 1] * df, h[t])
    # MCS estimator
    C0 = df * np.mean(V[1])
    return C0


if __name__ == '__main__':
    M = 50
    S0 = 100
    r = 0.05
    sigma = 0.25
    T = 2.0
    I = 10000
    # ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * npr.standard_normal(I))
    # MCS1 = montacarlo_simulation()
    # BM1 = geometric_brownian_motion()
    # SQRT_DIFF = square_root_diffusion_euler()
    # print_statistics(ST1, MCS1)
    # print_statistics(ST1, BM1[-1])
    # stochastic_volatility()

    # jump_diffusion()

    # print(gbm_mcs_stat(K=105.))
    # print(gbm_mcs_dyna(K=105., option='call'))
    print(gbm_mcs_amer(K=105., option='put'))

    # print(gbm_mcs_stat(K=105.))
    # print(gbm_mcs_dyna(K=105., option='put'))
    # print(gbm_mcs_amer(K=105., option='put'))
