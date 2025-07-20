import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib


class LinearStockPredictor:
    def __init__(self, T=60, T_prime=10):
        self.T = T
        self.T_prime = T_prime
        self.model = None
        self.scaler = None
        self.df = None

    def load_and_preprocess(self, df):
        self.df = df.copy()
        self.df.sort_values('Date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        close = self.df['Close'].values.reshape(-1, 1)
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(close)

        X, y = [], []
        for i in range(len(scaled) - self.T):
            X.append(scaled[i:i + self.T].flatten())
            y.append(scaled[i + self.T][0])
        return np.array(X), np.array(y)

    def train(self, df):
        X, y = self.load_and_preprocess(df)
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict_future(self):
        if self.model is None or self.scaler is None or self.df is None:
            raise ValueError("Model must be trained first.")

        close = self.df['Close'].values.reshape(-1, 1)
        scaled = self.scaler.transform(close)
        last_seq = scaled[-self.T:].flatten().reshape(1, -1)

        preds_scaled = []
        for _ in range(self.T_prime):
            pred = self.model.predict(last_seq)[0]
            preds_scaled.append(pred)
            last_seq = np.append(last_seq[:, 1:], [[pred]], axis=1)

        preds = self.scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(
            start=self.df['Date'].max() + pd.Timedelta(days=1),
            periods=self.T_prime,
            freq='B'
        )

        # ✅ Trả về DataFrame có cột 'Date' và 'Predicted'
        return pd.DataFrame({
        'Date': future_dates,
        'Prediction': preds  # ✅ đổi tên cột tại đây
        })


    def save(self, path='linear_model.pkl'):
        joblib.dump((self.model, self.scaler, self.T, self.T_prime), path)

    def load(self, path='linear_model.pkl'):
        self.model, self.scaler, self.T, self.T_prime = joblib.load(path)
