import pandas as pd
import joblib
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc


class ProphetPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = self.load_model()
        self.forecast = None

    def load_model(self):
        if self.model_path is None:
            raise ValueError("Bạn chưa cung cấp model_path.")
        return joblib.load(self.model_path)

    def predict(self, historical_data: pd.DataFrame, period: int = 10):
        """
        Dự đoán với mô hình Prophet và trả về DataFrame với cột ['Date', 'Prediction']
        """
        df_train = historical_data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        future = self.model.make_future_dataframe(periods=period)
        forecast = self.model.predict(future)
        self.forecast = forecast

        # Trả về định dạng chuẩn hóa cho các hàm plot phía sau
        result_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Prediction'})
        return result_df

    def plot_prophet_forecast_percent_change(self, prediction_df: pd.DataFrame, ticker: str):
        """
        Vẽ biểu đồ phần trăm thay đổi giá dự đoán từ mô hình Prophet.
        """
        if prediction_df is None or prediction_df.empty:
            st.warning("Không có dữ liệu dự đoán để hiển thị.")
            return

        df = prediction_df.copy()
        if not set(['Date', 'Prediction']).issubset(df.columns):
            st.error("DataFrame không chứa cột 'Date' và 'Prediction'.")
            return

        df = df[['Date', 'Prediction']].sort_values("Date").reset_index(drop=True)

        # Lọc phần dự đoán tương lai nếu có lịch sử
        if hasattr(self.model, "history"):
            last_train_date = self.model.history['ds'].max()
            df = df[df['Date'] > last_train_date]

        if df.empty:
            st.warning("Không có phần dự đoán nào sau ngày cuối cùng của dữ liệu huấn luyện.")
            return

        initial_price = df['Prediction'].iloc[0]
        pct_change = ((df['Prediction'] / initial_price) - 1) * 100

        fig = go.Figure()
        colors = pc.qualitative.Vivid

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=pct_change,
            name=ticker,
            line=dict(color=colors[0], width=2)
        ))

        fig.update_layout(
            title=f"Thay đổi tỷ lệ dự đoán ({ticker}) theo thời gian",
            xaxis_title="Ngày",
            yaxis_title="Thay đổi (%)",
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)


    def plot_full_forecast(self, n_days: int, ticker: str):
        """
        Vẽ biểu đồ dự báo tổng thể và các thành phần xu hướng bằng Plotly (hỗ trợ dark theme).
        """
        if self.forecast is None:
            st.warning("Chưa có dữ liệu dự đoán để hiển thị.")
            return

        # Vẽ biểu đồ tổng thể
        fig1 = plot_plotly(self.model, self.forecast)
        fig1.update_layout(
            title=f"Biểu đồ dự đoán tổng thể với Prophet ({ticker}, {n_days} ngày)",
            xaxis_title="Ngày",
            yaxis_title="Giá dự đoán (USD)",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Vẽ thành phần xu hướng
        # Dựng từng biểu đồ trend/weekly/yearly bằng Plotly thủ công
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=self.model.history['ds'],
            y=self.model.history['y'],
            mode='lines',
            name='Dữ liệu gốc'
        ))
        trend_fig.add_trace(go.Scatter(
            x=self.forecast['ds'],
            y=self.forecast['trend'],
            mode='lines',
            name='Xu hướng'
        ))
        trend_fig.update_layout(
            title="Các thành phần xu hướng trong mô hình Prophet",
            xaxis_title="Ngày",
            yaxis_title="Trend",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(trend_fig, use_container_width=True)

        # Nếu cần, bạn có thể vẽ seasonal_weekly, seasonal_yearly tương tự

