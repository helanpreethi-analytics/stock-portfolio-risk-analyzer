import matplotlib.pyplot as plt
import seaborn as sns


def plot_price_signals(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"], label="Price", alpha=0.6)
    plt.plot(df["SMA_short"], label="Short SMA")
    plt.plot(df["SMA_long"], label="Long SMA")
    buys = df[df["Position"] == 1]
    sells = df[df["Position"] == -1]
    plt.scatter(buys.index, buys["Close"], marker="^", color="green", s=100, label="Buy")
    plt.scatter(sells.index, sells["Close"], marker="v", color="red", s=100, label="Sell")
    plt.title(f"{ticker} - Strategy Signals")
    plt.legend()
    plt.show()


def plot_correlation(returns_df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(returns_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Asset Correlation Matrix")
    plt.show()
