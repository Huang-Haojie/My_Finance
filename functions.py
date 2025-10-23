import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.optimize import minimize


def bsm_pricer(S0, K, r, y, sigma, T, option_type):
    d1 = (np.log(S0 / K) + (r - y + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    if option_type == 'c':
        return S0 * np.exp(-y * T) * Nd1 - K * np.exp(-r * T) * Nd2
    elif option_type == 'p':
        return K * np.exp(-r * T) * (1 - Nd2) - S0 * np.exp(-y * T) * (1 - Nd1)
    else:
        raise ValueError("Invalid option type, use 'c' for call or 'p' for put")

def bsm_ivol(S, K, r, y, T, option_type, f, x0):
    func = lambda sigma: bsm_pricer(S, K, r, y, sigma, T, option_type) - f
    return fsolve(func, x0)[0]

def bsm_greeks(S0, K, r, sigma, T, option_type):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    gamma = nd1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * np.sqrt(T) * nd1
    theta = (-0.5 * S0 * nd1 * sigma / np.sqrt(T) - r * K * np.exp(-r * T) * Nd2) if option_type == 'c' else (-0.5 * S0 * nd1 * sigma / np.sqrt(T) + r * K * np.exp(-r * T) * (1 - Nd2))
    rho = (K * T * np.exp(-r * T) * Nd2) if option_type == 'c' else (-K * T * np.exp(-r * T) * (1 - Nd2))
    return gamma, vega, theta, rho


def calculate_covariance_matrix(price_data, annualize=True):
    returns = price_data.pct_change().dropna()
    cov_matrix = returns.cov()
    if annualize:
        cov_matrix = cov_matrix * 252  # 252 trading days per year
    return cov_matrix

def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights, np.dot(cov_matrix, weights))

def calculate_portfolio_std(weights, cov_matrix):
    return np.sqrt(calculate_portfolio_variance(weights, cov_matrix))

def calculate_portfolio_sharpe(weights, returns, risk_free_rate=0.02):
    portfolio_returns = (returns+1).prod()-1
    portfolio_return = np.dot(weights, portfolio_returns)
    #print(portfolio_return)
    portfolio_std = calculate_portfolio_std(weights, portfolio_return)

    return (portfolio_return - risk_free_rate) / portfolio_std

def optimize_minimum_variance(returns):
    cov_matrix = returns.cov()
    n_assets = len(returns.columns)

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1.0/n_assets] * n_assets)

    result = minimize(
        calculate_portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds, constraints=constraints
    )
    return result.x


def optimize_maximum_sharpe(returns, risk_free_rate=0.02):
    n_assets = len(returns.columns)

    def negative_sharpe(weights, returns, risk_free_rate):
        return -calculate_portfolio_sharpe(weights, returns, risk_free_rate)

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1.0/n_assets] * n_assets)

    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(returns,risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

if __name__ == "__main__":
    S = 240.56
    k = 360
    r = 0.03986
    y = 0
    sigma = 0.5704
    T = 0.043835616438356165
    option_type = 'c'
    #print(bsm_greeks(S0, k, r, sigma, T, option_type))
    #print(bsm_pricer(S0, k, r, y, sigma, T, option_type))
    print(bsm_ivol(S, k, r, y, T, 'c', 0.24,0.2))
