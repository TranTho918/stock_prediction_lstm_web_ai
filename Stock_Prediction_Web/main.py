import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from datetime import date, timedelta
import yfinance as yf
import os

# Import các module đã tách riêng
from data_handler import (
    load_historical_data,
    get_stock_info
)
from lstm import Predict
from linear import LinearStockPredictor
from prophet_model import ProphetPredictor

# --- THIẾT LẬP TRANG (PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN) ---
st.set_page_config(page_title="Stock Wise - Cổ Phiếu Thông Minh", layout="wide", page_icon="📈")

# --- TIÊU ĐỀ CHÍNH ---
st.markdown(
    """
    <div style='text-align: center; margin-top: -60px; margin-bottom: 5px; background-color: transparent ; padding: 10px 0;'>
        <h1 style='font-size: 36px; color: #FF4B4B; font-family: "Segoe UI", "Roboto", "Arial", sans-serif; font-weight: 700; letter-spacing: 1px;'>
             STOCK WISE - Cổ Phiếu Thông Minh
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Khởi tạo session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'time_range' not in st.session_state:
    st.session_state.time_range = "1 năm"
if 'comparison_stocks' not in st.session_state:
    st.session_state.comparison_stocks = []
if 'comparison_page' not in st.session_state:
    st.session_state.comparison_page = 0
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'prediction_time_range' not in st.session_state:
    st.session_state.prediction_time_range = "TỐI ĐA"


# --- CÁC HÀM VẼ BIỂU ĐỒ ---

def plot_single_stock(data_to_plot):
    """Vẽ biểu đồ chuỗi thời gian cho một cổ phiếu."""
    fig = go.Figure()
    line_color = 'limegreen' if not data_to_plot.empty and data_to_plot['Close'].iloc[-1] >= data_to_plot['Close'].iloc[
        0] else 'tomato'
    fill_color = 'rgba(152, 251, 152, 0.1)' if line_color == 'limegreen' else 'rgba(255, 99, 71, 0.1)'
    fig.add_trace(go.Scatter(
        x=data_to_plot['Date'],
        y=data_to_plot['Close'],
        name='Giá Đóng cửa',
        line=dict(color=line_color, width=2),
        fill='tozeroy',
        fillcolor=fill_color
    ))
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="Giá (USD)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_candlestick(data_to_plot, ticker_name):
    """Vẽ biểu đồ nến cho một cổ phiếu."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data_to_plot['Date'],
        open=data_to_plot['Open'],
        high=data_to_plot['High'],
        low=data_to_plot['Low'],
        close=data_to_plot['Close'],
        name='Biến động giá'
    ))
    fig.update_layout(
        title=f'Biểu đồ Nến cho {ticker_name}',
        yaxis_title='Giá cổ phiếu (USD)',
        xaxis_title='Thời gian',
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_comparison(main_ticker, main_data, comparison_tickers, start_date, end_date):
    """Vẽ biểu đồ so sánh hiệu suất của nhiều cổ phiếu."""
    fig = go.Figure()
    all_tickers = [main_ticker] + comparison_tickers
    all_data = {main_ticker: main_data}
    colors = plotly.colors.qualitative.Plotly
    for ticker in comparison_tickers:
        all_data[ticker] = load_historical_data(ticker)
    for i, ticker in enumerate(all_tickers):
        df = all_data.get(ticker)
        if df is None:
            continue
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]
        if not filtered_df.empty:
            initial_price = filtered_df['Close'].iloc[0]
            pct_change = ((filtered_df['Close'] / initial_price) - 1) * 100
            fig.add_trace(go.Scatter(
                x=filtered_df['Date'],
                y=pct_change,
                name=ticker,
                line=dict(color=colors[i % len(colors)])
            ))
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="Thay đổi (%)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    return all_data


# --- CÁC HÀM HIỂN THỊ GIAO DIỆN ---

def display_comparison_legend(all_data, start_date, end_date):
    """Hiển thị chú thích và thông tin chi tiết cho biểu đồ so sánh."""
    st.markdown("---")
    colors = plotly.colors.qualitative.Plotly
    for i, (ticker, df) in enumerate(all_data.items()):
        if df is None:
            continue
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]
        if not filtered_df.empty:
            initial_price = filtered_df['Close'].iloc[0]
            final_price = filtered_df['Close'].iloc[-1]
            abs_change = final_price - initial_price
            pct_change = (abs_change / initial_price) * 100
            color = colors[i % len(colors)]
            change_color = "limegreen" if pct_change >= 0 else "tomato"
            change_icon = "🔼" if pct_change >= 0 else "🔽"
            cols = st.columns([0.5, 2, 2, 2, 2, 1])
            with cols[0]:
                st.markdown(f'<div style="width: 10px; height: 20px; background-color: {color};"></div>',
                            unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"**{ticker}**")
            with cols[2]:
                st.markdown(f"{final_price:.2f} $")
            with cols[3]:
                st.markdown(f"<span style='color:{change_color}'>{abs_change:+.2f} $</span>", unsafe_allow_html=True)
            with cols[4]:
                st.markdown(f"<span style='color:{change_color}'>{change_icon} {pct_change:+.2f}%</span>",
                            unsafe_allow_html=True)
            with cols[5]:
                if ticker != st.session_state.selected_stock:
                    if st.button("✕", key=f"remove_{ticker}"):
                        st.session_state.comparison_stocks.remove(ticker)
                        st.rerun()


def display_stock_info_card(info):
    """Hiển thị thẻ thông tin chi tiết của cổ phiếu."""
    st.subheader("Thống kê")

    def format_large_number(num):
        if num is None or num == 0: return "N/A"
        if num > 1_000_000_000_000: return f"{num / 1_000_000_000_000:.2f} NT"
        if num > 1_000_000_000: return f"{num / 1_000_000_000:.2f} Tỷ"
        if num > 1_000_000: return f"{num / 1_000_000:.2f} Tr"
        return f"{num:,}"

    html_string = """<style>.info-card table{width:100%;border-collapse:collapse;}.info-card td{padding:8px 5px;border-bottom:1px solid #31333F;font-size:14px;}.info-card .label{color:#A0A0A0;text-align:left;}.info-card .value{color:#FAFAFA;text-align:right;font-weight:600;}</style><div class="info-card"><table>"""
    info_rows = {
        "Giá đóng cửa hôm trước": f"{info.get('previousClose'):.2f} $" if info.get('previousClose') else "N/A",
        "Mức chênh lệch một ngày": f"{info.get('dayLow'):.2f} - {info.get('dayHigh'):.2f} $" if info.get(
            'dayLow') and info.get('dayHigh') else "N/A",
        "Phạm vi một năm": f"{info.get('fiftyTwoWeekLow'):.2f} - {info.get('fiftyTwoWeekHigh'):.2f} $" if info.get(
            'fiftyTwoWeekLow') and info.get('fiftyTwoWeekHigh') else "N/A",
        "Giá trị vốn hóa": f"{format_large_number(info.get('marketCap'))} USD",
        "KL giao dịch TB": format_large_number(info.get('averageVolume')),
        "Tỷ số P/E": f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A",
        "Tỷ lệ cổ tức": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A",
        "Sàn giao dịch": info.get('exchange', "N/A")
    }
    for label, value in info_rows.items():
        html_string += f'<tr><td class="label">{label}</td><td class="value">{value}</td></tr>'
    html_string += "</table></div>"
    st.markdown(html_string, unsafe_allow_html=True)


def display_about_section(info):
    """Hiển thị phần giới thiệu ngắn và thông tin chi tiết về công ty."""
    st.subheader("Giới thiệu")
    summary = info.get('longBusinessSummary')
    if summary:
        st.markdown(f"""
        <div style="text-align: justify; font-size: 14px; color: #A0A0A0;">
            {summary[:400]}...
            <a href="https://vi.wikipedia.org/wiki/{info.get('shortName', '').replace(' ', '_')}" target="_blank">Wikipedia</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Không có thông tin giới thiệu cho mã này.")
    st.write("")
    ceo_name = "N/A"
    if info.get('companyOfficers'):
        for officer in info['companyOfficers']:
            if 'ceo' in officer.get('title', '').lower():
                ceo_name = officer.get('name', 'N/A')
                break
    address = f"{info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}".strip(", ")
    website = info.get('website', 'N/A')
    employees = f"{info.get('fullTimeEmployees'):,}" if info.get('fullTimeEmployees') else "N/A"
    html_string = """<style>.about-card table{width:100%;border-collapse:collapse;}.about-card td{padding:8px 5px;border-bottom:1px solid #31333F;font-size:14px;}.about-card .label{color:#A0A0A0;text-align:left;}.about-card .value{color:#FAFAFA;text-align:right;font-weight:600;}</style><div class="about-card"><table>"""
    about_rows = {
        "👤 Giám đốc điều hành": ceo_name,
        "📍 Trụ sở chính": address,
        "🌐 Trang web": f"<a href='http://{website}' target='_blank'>{website}</a>",
        "👥 Nhân viên": employees
    }
    for label, value in about_rows.items():
        html_string += f'<tr><td class="label">{label}</td><td class="value">{value}</td></tr>'
    html_string += "</table></div>"
    st.markdown(html_string, unsafe_allow_html=True)


def display_main_header(info, data, time_range_label, start_date_value):
    """Hiển thị phần header chính với giá và thay đổi theo bộ lọc thời gian."""
    stock_name = info.get('longName', info.get('shortName', ''))
    if stock_name:
        st.markdown(f"<h2 style='font-weight: 600; margin-bottom: 5px;'>{stock_name}</h2>", unsafe_allow_html=True)
        st.divider()
    price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    change = 0
    change_pct = 0
    if data is not None and not data.empty:
        period_data = data[data['Date'] >= pd.to_datetime(start_date_value)]
        if len(period_data) > 1:
            start_price = period_data['Close'].iloc[0]
            end_price = period_data['Close'].iloc[-1]
            change = end_price - start_price
            change_pct = (change / start_price) if start_price != 0 else 0
    color = "limegreen" if change >= 0 else "tomato"
    change_sign = "+" if change >= 0 else ""
    icon = "🔼" if change >= 0 else "🔽"
    st.markdown(f"""
    <div style="display: flex; align-items: baseline; gap: 15px; margin-bottom: 5px;">
        <span style="font-size: 48px; font-weight: bold;">{price:.2f} $</span>
        <span style="font-size: 24px; color: {color}; font-weight: bold;">{icon} {change_pct * 100:.2f}%</span>
        <span style="font-size: 16px; color: {color};">{change_sign}{change:.2f} {time_range_label}</span>
    </div>
    """, unsafe_allow_html=True)
    pre_price = info.get('preMarketPrice')
    if pre_price:
        pre_change = info.get('preMarketChange', 0)
        pre_change_pct = info.get('preMarketChangePercent', 0)
        pre_color = "limegreen" if pre_change >= 0 else "tomato"
        pre_change_icon = "🔼" if pre_change >= 0 else "🔽"
        st.markdown(f"""
        <div style="font-size: 14px; color: #A0A0A0;">
            <span>Trước giờ mở cửa: </span>
            <span style="color: #FAFAFA; font-weight: 600;">{pre_price:.2f} $</span>
            <span style="color: {pre_color};"> ({pre_change_icon} {pre_change:+.2f} {pre_change_pct * 100:+.2f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    st.write("")


def plot_prediction_results(historical_data, prediction_results, time_range_label="1 tháng"):
    fig = go.Figure()
    today = date.today()
    if time_range_label == "1 ngày":
        start_date_filter = today - timedelta(days=2)
    elif time_range_label == "5 ngày":
        start_date_filter = today - timedelta(days=7)
    elif time_range_label == "1 tháng":
        start_date_filter = today - timedelta(days=31)
    elif time_range_label == "6 tháng":
        start_date_filter = today - timedelta(days=183)
    elif time_range_label == "Từ đầu năm":
        start_date_filter = date(today.year, 1, 1)
    elif time_range_label == "1 năm":
        start_date_filter = today - timedelta(days=366)
    elif time_range_label == "5 năm":
        start_date_filter = today - timedelta(days=5 * 366)
    else:
        start_date_filter = historical_data['Date'].min()
    hist_filtered = historical_data[historical_data['Date'] >= pd.to_datetime(start_date_filter)]
    fig.add_trace(go.Scatter(
        x=hist_filtered['Date'],
        y=hist_filtered['Close'],
        mode='lines',
        name='Dữ liệu Lịch sử',
        line=dict(color='gray')
    ))
    colors = plotly.colors.qualitative.Vivid
    for i, (model_name, pred_df) in enumerate(prediction_results.items()):
        pred_filtered = pred_df[pred_df['Date'] >= pd.to_datetime(start_date_filter)]
        if not pred_filtered.empty:
            last_hist_date = historical_data['Date'].max()
            last_hist_price = historical_data.loc[historical_data['Date'] == last_hist_date, 'Close'].values[0]
            pred_filtered = pd.concat([
                pd.DataFrame({'Date': [last_hist_date], 'Prediction': [last_hist_price]}),
                pred_filtered
            ], ignore_index=True)
            fig.add_trace(go.Scatter(
                x=pred_filtered['Date'],
                y=pred_filtered['Prediction'],
                mode='lines',
                name=f'Dự đoán ({model_name})',
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))
    fig.update_layout(
        title="Biểu đồ Dự đoán Giá Cổ phiếu",
        xaxis_title="Ngày",
        yaxis_title="Giá (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark',
        height=450,
        margin=dict(t=40, b=30, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)


# --- GIAO DIỆN CHÍNH CỦA ỨNG DỤNG ---

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { display: flex; justify-content: space-around; }
    .stTabs [data-baseweb="tab"] { flex-grow: 1; text-align: center; }
    .stTabs [data-baseweb="tab"] button { font-size: 16px !important; font-weight: 600; color: #FAFAFA; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] button { color: #FF4B4B; }
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "PHÂN TÍCH & DỰ BÁO",
    "TIN TỨC",
    "CHATBOT HỖ TRỢ"
])

with tab1:
    st.header("1. Chọn một mã cổ phiếu")
    stocks = ("AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM")


    def handle_stock_selection(stock):
        st.session_state.selected_stock = stock
        st.session_state.time_range = "1 năm"
        st.session_state.comparison_stocks = []
        st.session_state.comparison_page = 0
        st.session_state.prediction_results = None


    N_COLS = 5
    cols = st.columns(N_COLS)
    for i, stock_ticker in enumerate(stocks):
        with cols[i % N_COLS]:
            is_selected = (stock_ticker == st.session_state.selected_stock)
            button_type = "primary" if is_selected else "secondary"
            st.button(stock_ticker, key=f"select_{stock_ticker}", on_click=handle_stock_selection, args=(stock_ticker,),
                      use_container_width=True, type=button_type)
    if st.session_state.selected_stock:
        selected_stock = st.session_state.selected_stock
        historical_data = load_historical_data(selected_stock)
        stock_info = get_stock_info(selected_stock)
        st.write("---")
        main_col, stats_col = st.columns([2.5, 1])
        with main_col:
            if historical_data is not None and not historical_data.empty:
                today = date.today()
                min_available_date = historical_data['Date'].min().date()
                time_range_options = {"1 ngày": 2, "5 ngày": 7, "1 tháng": 31, "6 tháng": 183,
                                      "Từ đầu năm": (today - date(today.year, 1, 1)).days, "1 năm": 366,
                                      "5 năm": 5 * 366}
                if st.session_state.time_range in time_range_options:
                    start_date_value = today - timedelta(days=time_range_options[st.session_state.time_range])
                else:  # TỐI ĐA
                    start_date_value = min_available_date
                if start_date_value < min_available_date:
                    start_date_value = min_available_date
                if stock_info:
                    display_main_header(stock_info, historical_data, st.session_state.time_range, start_date_value)
                else:
                    st.header(f"Phân tích lịch sử giá của {selected_stock}")
                time_ranges = ["1 ngày", "5 ngày", "1 tháng", "6 tháng", "Từ đầu năm", "1 năm", "5 năm", "TỐI ĐA"]
                st.radio("Chọn nhanh khoảng thời gian", options=time_ranges, key='time_range', horizontal=True,
                         label_visibility="collapsed")
                filtered_data = historical_data[historical_data['Date'] >= pd.to_datetime(start_date_value)]

                # --- THAY ĐỔI: GỘP BIỂU ĐỒ ---
                if st.session_state.comparison_stocks:
                    # Nếu đang so sánh, chỉ hiển thị biểu đồ so sánh
                    comparison_data = plot_comparison(selected_stock, historical_data,
                                                      st.session_state.comparison_stocks, start_date_value, today)
                    display_comparison_legend(comparison_data, start_date_value, today)
                else:
                    # Nếu xem một mã, cho phép chuyển đổi
                    chart_type = st.radio(
                        "Chọn loại biểu đồ:",
                        ('Biểu đồ đường', 'Biểu đồ nến'),
                        horizontal=True,
                        key='chart_type_selector'
                    )

                    if chart_type == 'Biểu đồ đường':
                        plot_single_stock(filtered_data)
                    else:
                        plot_candlestick(filtered_data, selected_stock)

                # --- KẾT THÚC THAY ĐỔI ---

                st.write("---")
                st.subheader("So sánh với")
                stocks_to_compare = [s for s in stocks if
                                     s != selected_stock and s not in st.session_state.comparison_stocks]
                if not stocks_to_compare:
                    st.info("Đã thêm tất cả các mã có sẵn để so sánh.")
                else:
                    items_per_page = 5
                    start_idx = st.session_state.comparison_page * items_per_page
                    stocks_on_page = stocks_to_compare[start_idx: start_idx + items_per_page]
                    compare_cols = st.columns(items_per_page + 1)
                    for i, ticker in enumerate(stocks_on_page):
                        with compare_cols[i]:
                            if st.button(ticker, key=f"compare_{ticker}", use_container_width=True):
                                st.session_state.comparison_stocks.append(ticker)
                                st.rerun()
                    with compare_cols[items_per_page]:
                        if st.button("▶", key="next_page"):
                            st.session_state.comparison_page = (st.session_state.comparison_page + 1) % (
                                        (len(stocks_to_compare) + items_per_page - 1) // items_per_page)
                            st.rerun()
            else:
                st.error(f"Không thể tải dữ liệu lịch sử cho {selected_stock}.")
        with stats_col:
            if stock_info:
                display_stock_info_card(stock_info)
                st.write("---")
                display_about_section(stock_info)
            else:
                st.warning(f"Không thể tải thông tin chi tiết cho {selected_stock}.")
        if 'filtered_data' in locals() and not filtered_data.empty and not st.session_state.comparison_stocks:
            filtered_data = filtered_data.sort_values(by="Date", ascending=False)
            st.subheader("Bảng dữ liệu lịch sử")
            st.dataframe(filtered_data, use_container_width=True, hide_index=True)
        if "run_prophet" not in st.session_state: st.session_state.run_prophet = False
        if "run_lstm_linear" not in st.session_state: st.session_state.run_lstm_linear = False
        st.write("---")
        st.header("3. Phân tích & Dự báo xu hướng cổ phiếu bằng mô hình Prophet")
        period_prophet = st.number_input("Số ngày dự đoán (Prophet):", min_value=1, max_value=365, value=10, step=1,
                                         key="pred_days_prophet")
        ticker = st.session_state.selected_stock
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "prophet_models", f"prophet_{ticker}.pkl")
        if os.path.exists(model_path):
            predictor1 = ProphetPredictor(model_path=model_path)
            if st.button(f"Chạy dự đoán với Prophet cho {period_prophet} ngày", key="run_prediction_prophet"):
                st.session_state.run_prophet = True
            if st.session_state.run_prophet:
                prediction_df = predictor1.predict(historical_data, period=period_prophet)
                predictor1.plot_prophet_forecast_percent_change(prediction_df, ticker)
                predictor1.plot_full_forecast(period_prophet, ticker)
        else:
            st.error(f"File mô hình cho {ticker} không tồn tại tại đường dẫn:\n{model_path}")
        st.write("---")
        st.header("4. Thiết lập và Chạy Dự đoán")
        period = st.number_input("Số ngày dự đoán:", min_value=1, max_value=365, value=10, step=1,
                                 key="pred_days_lstm_linear")
        st.subheader("Chọn mô hình dự đoán để sử dụng. Bạn có thể chọn nhiều mô hình cùng lúc.")
        model_options = ("LSTM", "Linear Regression")
        cols = st.columns(len(model_options))
        selected_models = [model_name for i, model_name in enumerate(model_options) if
                           cols[i].checkbox(model_name, key=f"model_{model_name}")]
        if st.button(f"Chạy dự đoán với các mô hình cho {period} ngày", key="run_prediction_lstm_linear"):
            if not selected_models:
                st.warning("Vui lòng chọn ít nhất một mô hình để chạy dự đoán.")
            else:
                with st.spinner(f"Đang dự đoán bằng mô hình: {', '.join(selected_models)}..."):
                    prediction_results = {}
                    for model_name in selected_models:
                        if model_name == "LSTM":
                            model_path = os.path.join(BASE_DIR, "model_lstm", f"lstm_{ticker}.h5")
                            if os.path.exists(model_path):
                                predictor = Predict(model_path=model_path)
                                prediction_results[model_name] = predictor.predict(historical_data, period)
                            else:
                                st.error(f"File mô hình LSTM cho {ticker} không tồn tại tại:\n{model_path}")
                        elif model_name == "Linear Regression":
                            linear_model = LinearStockPredictor(T=60, T_prime=period)
                            linear_model.train(historical_data)
                            prediction_results[model_name] = linear_model.predict_future()
                    st.session_state.prediction_results = prediction_results
                st.success(f"Đã chạy dự đoán với mô hình: {', '.join(selected_models)}")
        if st.session_state.prediction_results:
            st.subheader("Kết quả Dự đoán")
            pred_time_ranges = ["1 ngày", "5 ngày", "1 tháng", "6 tháng", "Từ đầu năm", "1 năm", "5 năm", "TỐI ĐA"]
            st.radio("Phóng to kết quả dự đoán", options=pred_time_ranges, key='pred_time_ranges', index=2,
                     horizontal=True, label_visibility="collapsed")
            plot_prediction_results(historical_data, st.session_state.prediction_results,
                                    st.session_state.pred_time_ranges)
    else:
        st.info("Vui lòng chọn một mã cổ phiếu để bắt đầu phân tích.")

# --- TAB 2: TIN TỨC ---
with tab2:
    st.header("📰 Tin tức thị trường")

    if 'news_data' not in st.session_state:
        st.session_state.news_data = None
    if 'last_news_stock' not in st.session_state:
        st.session_state.last_news_stock = None


    def fetch_news():
        if st.session_state.selected_stock:
            with st.spinner('Đang tải tin tức...'):
                try:
                    ticker_obj = yf.Ticker(st.session_state.selected_stock)
                    st.session_state.news_data = ticker_obj.news
                    st.session_state.last_news_stock = st.session_state.selected_stock
                except Exception as e:
                    st.session_state.news_data = {"error": str(e)}


    # Tự động tải tin tức khi cổ phiếu thay đổi hoặc khi tab được tải lần đầu
    if st.session_state.selected_stock and st.session_state.selected_stock != st.session_state.last_news_stock:
        fetch_news()

    # Giao diện
    if st.session_state.selected_stock:
        st.subheader(f"Tin tức mới nhất về {st.session_state.selected_stock}")

        if st.button("Làm mới tin tức", key="refresh_news"):
            fetch_news()

        news_list = st.session_state.get('news_data')

        if news_list is None:
            st.info("Nhấn 'Làm mới tin tức' để tải.")
        elif isinstance(news_list, dict) and 'error' in news_list:
            st.error(f"Đã xảy ra lỗi khi cố gắng tải tin tức: {news_list['error']}")
            st.info("API tin tức có thể đang gặp sự cố tạm thời. Vui lòng thử lại sau.")
        elif not news_list:
            st.warning(f"Không tìm thấy tin tức nào cho {st.session_state.selected_stock}.")
            st.info("API của Yahoo Finance có thể không cung cấp tin tức cho mã này.")
        else:
            displayed_count = 0
            for item in news_list:
                content = item.get('content', {})
                title = content.get('title')

                canonical_url = content.get('canonicalUrl', {})
                link = canonical_url.get('url') if isinstance(canonical_url, dict) else content.get('clickThroughUrl',
                                                                                                    {}).get('url')

                if not link:
                    click_url = content.get('clickThroughUrl', {})
                    link = click_url.get('url') if isinstance(click_url, dict) else None

                if title and link:
                    displayed_count += 1
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 4])

                        provider = content.get('provider', {})
                        publisher = provider.get('displayName', 'Không rõ nguồn')
                        publish_time = content.get('pubDate')

                        thumbnail_url = None
                        thumbnail = content.get('thumbnail', {})
                        resolutions = thumbnail.get('resolutions')
                        if resolutions and isinstance(resolutions, list) and len(resolutions) > 0:
                            thumbnail_url = resolutions[0].get('url')

                        if thumbnail_url:
                            with col1:
                                st.image(thumbnail_url)

                        with col2:
                            st.subheader(f"[{title}]({link})")
                            time_str = pd.to_datetime(publish_time).strftime('%Y-%m-%d %H:%M') if publish_time else ""
                            st.write(f"**Nguồn:** {publisher} | **Thời gian:** {time_str}")

            if displayed_count == 0 and news_list:
                st.warning("Đã nhận được dữ liệu từ API nhưng không có tin tức nào có đủ định dạng để hiển thị.")
    else:
        st.info("Vui lòng chọn một mã cổ phiếu ở tab 'Phân tích & Dự báo' để xem tin tức.")

# --- TAB 3: CHATBOT ---
with tab3:
    st.header("🤖 Trợ lý AI dự báo xu hướng cổ phiếu")
    st.markdown("Hỏi tôi về xu hướng cổ phiếu, chỉ báo kỹ thuật, hoặc cách sử dụng hệ thống dự báo.")
    botpress_iframe_url = "https://cdn.botpress.cloud/webchat/v3.1/shareable.html?configUrl=https://files.bpcontent.cloud/2025/07/17/04/20250717042032-HACMEXE8.json"
    st.components.v1.iframe(botpress_iframe_url, height=600, scrolling=True)