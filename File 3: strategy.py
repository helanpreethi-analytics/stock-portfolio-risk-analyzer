class Strategy:
    def __init__(self, data):
        self.data = data

    def generate_signals(self):
        raise NotImplementedError


class MovingAverageCrossover(Strategy):
    def __init__(self, data, short_window=20, long_window=50):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        df = self.data.copy()
        df["SMA_short"] = df["Close"].rolling(self.short_window).mean()
        df["SMA_long"] = df["Close"].rolling(self.long_window).mean()
        df["Signal"] = 0
        df.loc[df["SMA_short"] > df["SMA_long"], "Signal"] = 1
        df["Position"] = df["Signal"].diff()
        return df
