import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import date, timedelta


@st.cache_data
def load_historical_data(ticker):
    """Tải và làm sạch dữ liệu lịch sử cho một mã cổ phiếu."""
    try:
        stock_ticker_obj = yf.Ticker(ticker)
        data = stock_ticker_obj.history(period="max")
        if data.empty: 
            return None

        # Xóa các cột không cần thiết
        cols_to_drop = ['Dividends', 'Stock Splits']
        for col in cols_to_drop:
            if col in data.columns: 
                data.drop(columns=[col], inplace=True)

        # Chuyển đổi kiểu dữ liệu
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns: 
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Xóa các dòng có giá trị NaN
        data.dropna(subset=['Close'], inplace=True)

        # Reset index và xử lý cột Date
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        
        return data
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu cho {ticker}: {e}")
        return None


@st.cache_data
def get_stock_info(ticker):
    """Tải thông tin chi tiết (info) cho một mã cổ phiếu."""
    try:
        return yf.Ticker(ticker).info
    except Exception as e:
        print(f"Lỗi khi tải thông tin cho {ticker}: {e}")
        return {}


def get_filtered_data(data, time_range):
    """Lọc dữ liệu theo khoảng thời gian."""
    if data is None or data.empty:
        return data
    
    today = date.today()
    min_available_date = data['Date'].min().date()
    
    if time_range == "1 ngày":
        start_date_value = today - timedelta(days=2)
    elif time_range == "5 ngày":
        start_date_value = today - timedelta(days=7)
    elif time_range == "1 tháng":
        start_date_value = today - timedelta(days=31)
    elif time_range == "6 tháng":
        start_date_value = today - timedelta(days=183)
    elif time_range == "Từ đầu năm":
        start_date_value = date(today.year, 1, 1)
    elif time_range == "1 năm":
        start_date_value = today - timedelta(days=366)
    elif time_range == "5 năm":
        start_date_value = today - timedelta(days=5 * 366)
    else:  # TỐI ĐA
        start_date_value = min_available_date
    
    # Đảm bảo start_date không nhỏ hơn ngày có dữ liệu đầu tiên
    if start_date_value < min_available_date:
        start_date_value = min_available_date
    
    return data[data['Date'] >= pd.to_datetime(start_date_value)]


def format_large_number(num):
    """Định dạng số lớn thành dạng dễ đọc."""
    if num is None or num == 0: 
        return "N/A"
    if num > 1_000_000_000_000: 
        return f"{num / 1_000_000_000_000:.2f} NT"
    if num > 1_000_000_000: 
        return f"{num / 1_000_000_000:.2f} Tỷ"
    if num > 1_000_000: 
        return f"{num / 1_000_000:.2f} Tr"
    return f"{num:,}"


def get_stock_statistics(info):
    """Tạo dictionary chứa các thống kê chính của cổ phiếu."""
    return {
        "Giá đóng cửa hôm trước": f"{info.get('previousClose'):.2f} $" if info.get('previousClose') else "N/A",
        "Mức chênh lệch một ngày": f"{info.get('dayLow'):.2f} - {info.get('dayHigh'):.2f} $" if info.get('dayLow') and info.get('dayHigh') else "N/A",
        "Phạm vi một năm": f"{info.get('fiftyTwoWeekLow'):.2f} - {info.get('fiftyTwoWeekHigh'):.2f} $" if info.get('fiftyTwoWeekLow') and info.get('fiftyTwoWeekHigh') else "N/A",
        "Giá trị vốn hóa": f"{format_large_number(info.get('marketCap'))} USD",
        "KL giao dịch TB": format_large_number(info.get('averageVolume')),
        "Tỷ số P/E": f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A",
        "Tỷ lệ cổ tức": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A",
        "Sàn giao dịch": info.get('exchange', "N/A")
    }


def get_company_details(info):
    """Tạo dictionary chứa thông tin chi tiết về công ty."""
    # Tìm CEO
    ceo_name = "N/A"
    if info.get('companyOfficers'):
        for officer in info['companyOfficers']:
            if 'ceo' in officer.get('title', '').lower():
                ceo_name = officer.get('name', 'N/A')
                break
    
    # Địa chỉ
    address = f"{info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}".strip(", ")
    
    # Website
    website = info.get('website', 'N/A')
    
    # Nhân viên
    employees = f"{info.get('fullTimeEmployees'):,}" if info.get('fullTimeEmployees') else "N/A"
    
    return {
        "👤 Giám đốc điều hành": ceo_name,
        "📍 Trụ sở chính": address,
        "🌐 Trang web": f"<a href='http://{website}' target='_blank'>{website}</a>",
        "👥 Nhân viên": employees
    }


def calculate_price_change(info, data, start_date_value):
    """Tính toán thay đổi giá theo khoảng thời gian."""
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
    
    return price, change, change_pct


def get_premarket_info(info):
    """Lấy thông tin giao dịch trước giờ mở cửa."""
    pre_price = info.get('preMarketPrice')
    if pre_price:
        pre_change = info.get('preMarketChange', 0)
        pre_change_pct = info.get('preMarketChangePercent', 0)
        return {
            'price': pre_price,
            'change': pre_change,
            'change_pct': pre_change_pct
        }
    return None