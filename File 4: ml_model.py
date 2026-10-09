from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def build_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df


def train_model(df):
    features = ["Return", "SMA10", "Volatility"]
    X = df[features]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return model, accuracy
