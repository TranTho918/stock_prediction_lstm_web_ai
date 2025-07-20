from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Predict:
    def __init__(self, model_path, time_step=60, T_prime=10):
        """
        Khởi tạo class Predict với đường dẫn model Keras và các tham số thời gian.

        Args:
            model_path (str): đường dẫn file model (.keras hoặc .h5)
            time_step (int): số bước đầu vào T
            T_prime (int): số bước đầu ra T'
        """
        self.model_path = model_path
        self.time_step = time_step
        self.T_prime = T_prime
        self.model = self.load_model()

    def load_model(self):
        """
        Load model Keras từ file .keras hoặc .h5
        """
        model = load_model(self.model_path, compile=False)
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, data, period):
        """
        Dự đoán chuỗi giá tương lai (dài T') từ dữ liệu đầu vào.
        Áp dụng lặp lại nhiều lần để dự đoán tổng cộng `period` ngày.

        Args:
            data (pd.DataFrame): dữ liệu gốc chứa 'Date' và 'Close'
            period (int): tổng số ngày cần dự đoán

        Returns:
            pd.DataFrame: chứa 'Date' và 'Prediction' (giá dự đoán)
        """
        df = data[['Date', 'Close']].copy()
        close_prices = df['Close'].values.astype(float).reshape(-1, 1)

        # Scale giá
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)

        # Chuỗi đầu vào ban đầu: lấy T giá cuối cùng
        test_inputs = scaled[-self.time_step:].reshape(-1).tolist()
        predictions_scaled = []

        steps = period // self.T_prime  # số lần lặp (mỗi lần dự đoán T' ngày)
        if period % self.T_prime != 0:
            steps += 1  # nếu dư, thêm 1 lần nữa

        for _ in range(steps):
            input_seq = np.array(test_inputs[-self.time_step:]).reshape(1, self.time_step, 1)
            pred = self.model.predict(input_seq, verbose=0)  # (1, T_prime, 1)
            pred_values = pred.reshape(-1).tolist()  # flatten về 1D list
            test_inputs.extend(pred_values)
            predictions_scaled.extend([[val] for val in pred_values])

        # Lấy đúng `period` kết quả đầu ra
        predictions_scaled = predictions_scaled[:period]

        # Inverse transform để ra giá gốc
        predictions = scaler.inverse_transform(predictions_scaled).flatten()

        # Sinh dãy ngày tương lai
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period)

        return pd.DataFrame({
            'Date': future_dates,
            'Prediction': predictions
        })

