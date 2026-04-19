import os
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# =========================
# CẤU HÌNH CHUNG
# =========================
CSV_PATH = "sales_data.csv"
MODEL_LOOKBACK = 14
DEFAULT_LEAD_TIME = 7
DEFAULT_SERVICE_LEVEL_Z = 1.65  # ~95%
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class InventoryResult:
    avg_daily_demand: float
    demand_std: float
    safety_stock: float
    reorder_point: float
    eoq: float


# =========================
# TẠO DỮ LIỆU MẪU NẾU CHƯA CÓ FILE
# =========================
def create_sample_data(path: str = CSV_PATH, days: int = 365) -> pd.DataFrame:
    """
    Tạo dữ liệu mẫu nếu chưa có sales_data.csv
    Cấu trúc file:
    date, sales, temperature, holiday
    """
    dates = pd.date_range(start="2025-01-01", periods=days, freq="D")

    trend = np.linspace(120, 170, days)
    weekly = 18 * np.sin(np.arange(days) * 2 * np.pi / 7)
    seasonality = 25 * np.sin(np.arange(days) * 2 * np.pi / 30)
    temperature = 30 + 3 * np.sin(np.arange(days) * 2 * np.pi / 365) + np.random.normal(0, 1.2, days)

    holiday = np.zeros(days)
    holiday[[40, 41, 42, 120, 121, 240, 241, 330, 331]] = 1

    noise = np.random.normal(0, 8, days)
    sales = trend + weekly + seasonality + holiday * 35 + (temperature - 30) * 1.3 + noise
    sales = np.maximum(sales, 20).round().astype(int)

    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "temperature": np.round(temperature, 1),
        "holiday": holiday.astype(int),
    })

    df.to_csv(path, index=False)
    return df


# =========================
# ĐỌC DỮ LIỆU
# =========================
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return create_sample_data(path)

    df = pd.read_csv(path)
    required_cols = {"date", "sales"}
    if not required_cols.issubset(df.columns):
        raise ValueError("File CSV phải có ít nhất 2 cột: date, sales")

    if "temperature" not in df.columns:
        df["temperature"] = 30
    if "holiday" not in df.columns:
        df["holiday"] = 0

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =========================
# TIỀN XỬ LÝ
# =========================
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    return data


def create_sequences(feature_array: np.ndarray, target_array: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(feature_array)):
        X.append(feature_array[i - lookback:i])
        y.append(target_array[i])
    return np.array(X), np.array(y)


# =========================
# TRAIN MODEL DỰ BÁO
# =========================
def train_forecast_model(df: pd.DataFrame, lookback: int = MODEL_LOOKBACK):
    data = make_features(df)

    feature_cols = ["sales", "temperature", "holiday", "day_of_week", "month", "is_weekend"]
    features = data[feature_cols].astype(float)
    target = data[["sales"]].astype(float)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)

    X, y = create_sequences(features_scaled, target_scaled, lookback)

    if len(X) < 10:
        raise ValueError("Dữ liệu quá ít để train model. Hãy tăng số dòng dữ liệu.")

    # Đổi từ 3D sang 2D để dùng LinearRegression
    X_flat = X.reshape(X.shape[0], -1)
    y_flat = y.ravel()

    split_idx = int(len(X_flat) * 0.8)
    X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
    y_train, y_test = y_flat[:split_idx], y_flat[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test_scaled)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "data": data,
        "mape": mape,
        "X_test": X_test,
        "y_true": y_true.flatten(),
        "y_pred": y_pred.flatten(),
        "lookback": lookback,
    }


# =========================
# DỰ BÁO TƯƠNG LAI
# =========================
def forecast_next_days(train_result: dict, days_ahead: int = 7) -> pd.DataFrame:
    model = train_result["model"]
    data = train_result["data"].copy()
    feature_scaler = train_result["feature_scaler"]
    target_scaler = train_result["target_scaler"]
    feature_cols = train_result["feature_cols"]
    lookback = train_result["lookback"]

    recent_data = data.tail(lookback).copy()
    forecast_rows = []

    for _ in range(days_ahead):
        seq_features = recent_data[feature_cols].astype(float)
        seq_scaled = feature_scaler.transform(seq_features)
        seq_flat = seq_scaled.reshape(1, -1)

        pred_scaled = model.predict(seq_flat).reshape(-1, 1)
        pred_sales = target_scaler.inverse_transform(pred_scaled)[0][0]
        pred_sales = max(0, float(pred_sales))

        next_date = recent_data["date"].max() + pd.Timedelta(days=1)
        next_day_of_week = next_date.dayofweek
        next_month = next_date.month
        next_is_weekend = int(next_day_of_week >= 5)

        last_temp = float(recent_data["temperature"].iloc[-1])
        next_temp = last_temp + np.random.normal(0, 0.5)
        next_holiday = 0

        new_row = pd.DataFrame({
            "date": [next_date],
            "sales": [pred_sales],
            "temperature": [next_temp],
            "holiday": [next_holiday],
            "day_of_week": [next_day_of_week],
            "month": [next_month],
            "is_weekend": [next_is_weekend],
        })

        forecast_rows.append(new_row[["date", "sales"]])
        recent_data = pd.concat([recent_data, new_row], ignore_index=True).tail(lookback)

    forecast_df = pd.concat(forecast_rows, ignore_index=True)
    forecast_df["sales"] = forecast_df["sales"].round(0)
    return forecast_df


# =========================
# TÍNH INVENTORY METRICS
# =========================
def calculate_inventory_metrics(
    historical_sales: pd.Series,
    forecast_sales: pd.Series,
    lead_time_days: int = DEFAULT_LEAD_TIME,
    z_value: float = DEFAULT_SERVICE_LEVEL_Z,
    order_cost: float = 500000,
    holding_cost: float = 12000,
) -> InventoryResult:
    avg_daily_demand = float(forecast_sales.mean())
    demand_std = float(historical_sales.tail(30).std(ddof=1)) if len(historical_sales) >= 2 else 0.0

    safety_stock = z_value * demand_std * math.sqrt(max(lead_time_days, 1))
    reorder_point = avg_daily_demand * lead_time_days + safety_stock

    annual_demand = avg_daily_demand * 365
    eoq = math.sqrt((2 * annual_demand * order_cost) / max(holding_cost, 1))

    return InventoryResult(
        avg_daily_demand=avg_daily_demand,
        demand_std=demand_std,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        eoq=eoq,
    )


# =========================
# BIỂU ĐỒ
# =========================
def plot_actual_vs_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    tail_df = df.tail(60)
    ax.plot(tail_df["date"], tail_df["sales"], label="Lịch sử bán hàng")
    ax.plot(forecast_df["date"], forecast_df["sales"], label="Dự báo")
    ax.set_title("Nhu cầu lịch sử và dự báo")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Số lượng")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_train_result(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label="Thực tế")
    ax.plot(y_pred, label="Dự đoán")
    ax.set_title("Kết quả dự đoán trên tập test")
    ax.set_xlabel("Mốc thời gian")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# =========================
# GIAO DIỆN STREAMLIT
# =========================
def main():
    st.set_page_config(page_title="SmartStock AI Demo", layout="wide")
    st.title("SmartStock AI – Dự báo nhu cầu và quản trị tồn kho")
    st.caption("Demo cho đề tài logistics: Forecasting + Safety Stock + Reorder Point + EOQ")

    st.sidebar.header("Cấu hình mô phỏng")
    lookback = st.sidebar.slider("Số ngày nhìn lại (lookback)", 7, 30, 14)
    forecast_days = st.sidebar.slider("Số ngày muốn dự báo", 7, 30, 7)
    lead_time_days = st.sidebar.slider("Lead time (ngày)", 1, 30, 7)
    z_value = st.sidebar.selectbox("Mức độ an toàn (Z-score)", [1.28, 1.65, 1.96, 2.33], index=1)
    order_cost = st.sidebar.number_input("Chi phí mỗi lần đặt hàng", value=500000, step=50000)
    holding_cost = st.sidebar.number_input("Chi phí lưu kho / đơn vị / năm", value=12000, step=1000)

    st.info(
        "Nếu chưa có file sales_data.csv, chương trình sẽ tự tạo dữ liệu mẫu để bạn demo ngay. "
        "Khi có dữ liệu thật, chỉ cần thay file CSV bằng dữ liệu của nhóm."
    )

    try:
        df = load_data(CSV_PATH)
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return

    st.subheader("1) Dữ liệu đầu vào")
    st.dataframe(df.tail(10), use_container_width=True)

    with st.spinner("Đang train mô hình dự báo..."):
        try:
            result = train_forecast_model(df, lookback=lookback)
        except Exception as e:
            st.error(f"Lỗi train model: {e}")
            return

    forecast_df = forecast_next_days(result, days_ahead=forecast_days)

    inv = calculate_inventory_metrics(
        historical_sales=df["sales"],
        forecast_sales=forecast_df["sales"],
        lead_time_days=lead_time_days,
        z_value=z_value,
        order_cost=order_cost,
        holding_cost=holding_cost,
    )

    st.subheader("2) Kết quả chính")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAPE", f"{result['mape']:.2f}%")
    c2.metric("Nhu cầu TB/ngày", f"{inv.avg_daily_demand:.0f}")
    c3.metric("Safety Stock", f"{inv.safety_stock:.0f}")
    c4.metric("Reorder Point", f"{inv.reorder_point:.0f}")
    c5.metric("EOQ", f"{inv.eoq:.0f}")

    st.subheader("3) Bảng dự báo")
    display_forecast = forecast_df.copy()
    display_forecast["sales"] = display_forecast["sales"].round(0).astype(int)
    st.dataframe(display_forecast, use_container_width=True)

    st.subheader("4) Biểu đồ")
    plot_actual_vs_forecast(df, forecast_df)
    plot_train_result(result["y_true"], result["y_pred"])

    st.subheader("5) Giải thích nhanh để thuyết trình")
    st.markdown(
        f"""
**Mô hình dùng gì?**  
- Dùng mô hình dự báo để học từ dữ liệu bán hàng lịch sử và dự báo nhu cầu trong **{forecast_days} ngày tới**.

**Tồn kho an toàn tính sao?**  
- Safety Stock = Z × độ lệch chuẩn nhu cầu × căn bậc hai của lead time.

**Điểm đặt hàng lại là gì?**  
- Reorder Point = nhu cầu trung bình × lead time + safety stock.

**EOQ dùng để làm gì?**  
- Ước lượng lượng đặt hàng kinh tế để giảm tổng chi phí đặt hàng và lưu kho.
        """
    )

    st.subheader("6) Mẫu dữ liệu CSV cần chuẩn bị")
    st.code(
        "date,sales,temperature,holiday\n"
        "2025-01-01,120,30,0\n"
        "2025-01-02,135,31,0\n"
        "2025-01-03,160,29,1",
        language="csv",
    )

    st.success("Xong. Bạn có thể dùng bản này để demo ý tưởng cho nhóm trước, rồi nâng cấp bằng dữ liệu thật sau.")


if __name__ == "__main__":
    main()
