from data_loader import load_data, calculate_returns
from metrics import annual_return, annual_volatility, sharpe_ratio, max_drawdown, value_at_risk
from strategy import MovingAverageCrossover
from ml_model import build_features, train_model
from visualize import plot_price_signals, plot_correlation

tickers = ["AAPL", "MSFT", "TSLA"]
prices = load_data(tickers, "2022-01-01", "2024-01-01")
returns = calculate_returns(prices)

print("=== FINDINGS ===")
for t in tickers:
    print(f"\n{t}")
    print(f"Annual Return: {annual_return(returns[t]):.2%}")
    print(f"Annual Volatility: {annual_volatility(returns[t]):.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio(returns[t]):.2f}")
    print(f"Max Drawdown: {max_drawdown(prices[t]):.2%}")
    print(f"95% VaR: {value_at_risk(returns[t]):.2%}")

    # Strategy
    stock_df = prices[[t]].rename(columns={t: "Close"})
    signals = MovingAverageCrossover(stock_df).generate_signals()
    plot_price_signals(signals, t)

    # Machine learning
    model, acc = train_model(build_features(stock_df))
    print(f"ML Up/Down Accuracy: {acc:.2%}")

plot_correlation(returns)
