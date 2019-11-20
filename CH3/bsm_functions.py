from math import log, sqrt, exp
from scipy import stats


def d_1(S0, K, T, r, sigma):
    _d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return _d1


def d_2(S0, K, T, r, sigma):
    _d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return _d2


def bsm_call_value(S0, K, T, r, sigma):
    '''Valuation of a European call option in BSM Model
    Analytical formula:
    Parameters
    ============
    S0 : float
        initials stock/index level
    K : float
        strike price
    T : float
        maturity date  ( in year fractions)
    r : float
        risk free short rate
    sigma : float
        volatility factor in diffusion term

    Returns
    ========
    value : float
        present value of european call option
    '''

    S0 = float(S0)
    d1 = d_1(S0, K, T, r, sigma)
    d2 = d_2(S0, K, T, r, sigma)
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
             - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    return value


def bsm_vega(S0, K, T, r, sigma):
    '''Vega of a European call option in BSM Model
       Analytical formula:
       Parameters
       ============
       S0 : float
           initials stock/index level
       K : float
           strike price
       T : float
           maturity date  ( in year fractions)
       r : float
           risk free short rate
       sigma : float
           volatility factor in diffusion term

       Returns
       ========
       vega : float
           partial derivative if BSM formula with respect to sigma
       '''
    S0 = float(S0)
    d1 = d_1(S0, K, T, r, sigma)
    vega = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(T)
    return vega


def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    '''implied vol estimate  of a European call option in BSM Model
          Analytical formula:
          Parameters
          ============
          S0 : float
              initials stock/index level
          K : float
              strike price
          T : float
              maturity date  ( in year fractions)
          r : float
              risk free short rate
          c0 :
          sigma_est : float
              initial sigma estimate
         it : integer
            number of iterations

          Returns
          ========
          sigma_est : float
             numerical estimate of implied vol
          '''
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0) / bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est