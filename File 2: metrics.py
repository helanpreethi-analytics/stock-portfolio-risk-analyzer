import numpy as np


def annual_return(returns):
    return returns.mean() * 252


def annual_volatility(returns):
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns, risk_free=0.02):
    return (annual_return(returns) - risk_free) / annual_volatility(returns)


def max_drawdown(prices):
    cumulative = (1 + prices.pct_change()).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def value_at_risk(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)
