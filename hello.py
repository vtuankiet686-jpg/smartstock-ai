import os
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

CSV_PATH = "sales_data.csv"
MODEL_LOOKBACK = 14
DEFAULT_LEAD_TIME = 7
DEFAULT_SERVICE_LEVEL_Z = 1.65
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class InventoryResult:
    avg_daily_demand: float
    demand_std: float
    safety_stock: float
    reorder_point: float
    eoq: float


def create_sample_data(path: str = CSV_PATH, days: int = 365) -> pd.DataFrame:
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


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"date", "sales"}
    if not required_cols.issubset(df.columns):
        raise ValueError("File CSV phải có ít nhất 2 cột: date, sales")

    if "temperature" not in df.columns:
        df["temperature"] = 30
    if "holiday" not in df.columns:
        df["holiday"] = 0

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").fillna(30)
    df["holiday"] = pd.to_numeric(df["holiday"], errors="coerce").fillna(0)

    df = df.dropna(subset=["date", "sales"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 20:
        raise ValueError("CSV cần tối thiểu khoảng 20 dòng dữ liệu hợp lệ để chạy dự báo.")

    return df


def load_data_from_path(path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return create_sample_data(path)

    df = pd.read_csv(path)
    return normalize_dataframe(df)


def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return load_data_from_path(CSV_PATH)

    try:
        df = pd.read_csv(uploaded_file)
        return normalize_dataframe(df)
    except Exception as e:
        raise ValueError(f"Không đọc được file upload: {e}")


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
        pred_sales = max(20, float(pred_sales))

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

def load_saved_lstm_artifacts():
    model = load_model("lstm_model.keras")
    feature_scaler = joblib.load("feature_scaler.pkl")
    target_scaler = joblib.load("target_scaler.pkl")

    with open("model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "lookback": config["lookback"],
        "feature_cols": config["feature_cols"],
        "target_col": config["target_col"],
    }
def forecast_with_saved_lstm(df: pd.DataFrame, days_ahead: int = 7):
    artifacts = load_saved_lstm_artifacts()

    model = artifacts["model"]
    feature_scaler = artifacts["feature_scaler"]
    target_scaler = artifacts["target_scaler"]
    lookback = artifacts["lookback"]
    feature_cols = artifacts["feature_cols"]

    data = df.copy()

    if "temperature" not in data.columns:
        data["temperature"] = 30
    if "holiday" not in data.columns:
        data["holiday"] = 0

    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    recent_window = data.tail(lookback).copy()
    forecast_rows = []

    for _ in range(days_ahead):
        window_features = recent_window[feature_cols].astype(float)
        window_scaled = feature_scaler.transform(window_features)
        X_input = window_scaled.reshape(1, lookback, len(feature_cols))

        pred_scaled = model.predict(X_input, verbose=0)
        pred_sales = target_scaler.inverse_transform(pred_scaled)[0][0]
        pred_sales = max(20, float(pred_sales))

        next_date = recent_window["date"].max() + pd.Timedelta(days=1)
        next_day_of_week = next_date.dayofweek
        next_month = next_date.month
        next_is_weekend = int(next_day_of_week >= 5)

        last_temp = float(recent_window["temperature"].iloc[-1])
        next_temp = last_temp
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

        forecast_rows.append({
            "date": next_date,
            "sales": round(pred_sales)
        })

        recent_window = pd.concat([recent_window, new_row], ignore_index=True).tail(lookback)

    forecast_df = pd.DataFrame(forecast_rows)
    return forecast_df
def evaluate_saved_lstm_on_df(df: pd.DataFrame):
    artifacts = load_saved_lstm_artifacts()

    model = artifacts["model"]
    feature_scaler = artifacts["feature_scaler"]
    target_scaler = artifacts["target_scaler"]
    lookback = artifacts["lookback"]
    feature_cols = artifacts["feature_cols"]

    data = df.copy()

    if "temperature" not in data.columns:
        data["temperature"] = 30
    if "holiday" not in data.columns:
        data["holiday"] = 0

    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["sales"]).reset_index(drop=True)

    features_scaled = feature_scaler.transform(data[feature_cols].astype(float))
    target_scaled = target_scaler.transform(data[["sales"]].astype(float))

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(features_scaled[i-lookback:i])
        y.append(target_scaled[i])

    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mape": mape
    }

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


def evaluate_inventory_status(current_stock: float, inv: InventoryResult, forecast_df: pd.DataFrame):
    next_7_days_demand = float(forecast_df["sales"].sum())
    stock_gap_vs_rop = current_stock - inv.reorder_point
    suggested_order_qty = max(0, math.ceil(inv.eoq if current_stock < inv.reorder_point else 0))

    if current_stock < inv.safety_stock:
        status = "Nguy hiểm"
        priority = "Cao"
        color = "error"
        message = "Tồn kho hiện tại thấp hơn mức tồn kho an toàn. Nguy cơ thiếu hàng rất cao."
    elif current_stock < inv.reorder_point:
        status = "Cần nhập hàng"
        priority = "Trung bình"
        color = "warning"
        message = "Tồn kho hiện tại đã thấp hơn điểm đặt hàng lại. Nên lên đơn nhập thêm."
    elif current_stock > next_7_days_demand * 1.5 + inv.safety_stock:
        status = "Nguy cơ dư hàng"
        priority = "Trung bình"
        color = "warning"
        message = "Tồn kho hiện tại khá cao so với nhu cầu dự báo. Có nguy cơ dư hàng hoặc quay vòng chậm."
    else:
        status = "An toàn"
        priority = "Thấp"
        color = "success"
        message = "Tồn kho đang ở mức hợp lý so với nhu cầu dự báo."

    return {
        "status": status,
        "priority": priority,
        "color": color,
        "message": message,
        "next_7_days_demand": next_7_days_demand,
        "stock_gap_vs_rop": stock_gap_vs_rop,
        "suggested_order_qty": suggested_order_qty,
        "days_of_cover": current_stock / inv.avg_daily_demand if inv.avg_daily_demand > 0 else 0,
    }

def plot_actual_vs_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame):
    tail_df = df.tail(30).copy()
    forecast_df = forecast_df.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tail_df["date"],
        y=tail_df["sales"],
        mode="lines+markers",
        name="Lịch sử bán hàng",
        line=dict(width=3, color="#1f77b4"),
        marker=dict(size=8, color="#1f77b4")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["sales"],
        mode="lines+markers",
        name="Dự báo",
        line=dict(width=3, dash="dash", color="#ff7f0e"),
        marker=dict(size=8, color="#ff7f0e")
    ))

    if len(forecast_df) > 0:
        fig.add_annotation(
            x=forecast_df["date"].iloc[0],
            y=max(
                float(tail_df["sales"].max()) if len(tail_df) > 0 else 0,
                float(forecast_df["sales"].max()) if len(forecast_df) > 0 else 0
            ),
            text="Bắt đầu dự báo",
            showarrow=True,
            arrowhead=2,
            font=dict(size=13),
            yshift=10
        )

    fig.update_layout(
        title="Xu hướng bán hàng và dự báo",
        xaxis_title="Ngày",
        yaxis_title="Số lượng",
        hovermode="x unified",
        dragmode="pan",
        title_font=dict(size=20),
        font=dict(size=13),
        height=500,
        margin=dict(l=20, r=20, t=90, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        xaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14),
            nticks=6
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14)
        )
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def plot_train_result(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(y_true))),
        y=y_true,
        mode="lines+markers",
        name="Thực tế",
        line=dict(width=3, color="#2ca02c"),
        marker=dict(size=7, color="#2ca02c")
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode="lines+markers",
        name="Dự đoán",
        line=dict(width=3, dash="dash", color="#d62728"),
        marker=dict(size=7, color="#d62728")
    ))

    fig.update_layout(
        title="So sánh dữ liệu thực tế và dự đoán trên tập test",
        xaxis_title="Mốc thời gian",
        yaxis_title="Sales",
        hovermode="x unified",
        dragmode="pan",
        title_font=dict(size=20),
        font=dict(size=13),
        height=500,
        margin=dict(l=20, r=20, t=90, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        xaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14),
            nticks=8
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14)
        )
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main():
    st.set_page_config(page_title="SmartStock AI Demo", layout="wide")
    st.title("SmartStock AI – Dự báo nhu cầu và quản trị tồn kho")
    st.caption("Forecast và kết quả đánh giá được sinh từ mô hình TensorFlow LSTM (50 units) train trên Google Colab.")

    st.sidebar.header("Cấu hình mô phỏng")
    lookback = st.sidebar.slider("Số ngày nhìn lại (lookback)", 7, 30, 14)
    forecast_days = st.sidebar.slider("Số ngày muốn dự báo", 7, 30, 7)
    st.sidebar.caption("Nếu upload file mới, app sẽ train lại nhanh theo file đó. Nếu không upload, app dùng bộ LSTM đã train sẵn.")
    lead_time_days = st.sidebar.slider("Lead time (ngày)", 1, 30, 7)
    z_value = st.sidebar.selectbox("Mức độ an toàn (Z-score)", [1.28, 1.65, 1.96, 2.33], index=1)
    order_cost = st.sidebar.number_input("Chi phí mỗi lần đặt hàng", value=500000, step=50000)
    holding_cost = st.sidebar.number_input("Chi phí lưu kho / đơn vị / năm", value=12000, step=1000)
    scenario = st.sidebar.selectbox(
        "Kịch bản mô phỏng",
        ["Bình thường", "Tết", "Mưa lũ HCMC"],
        index=0
)

    st.subheader("1) Upload dữ liệu CSV")
    uploaded_file = st.file_uploader(
        "Tải file CSV của bạn lên",
        type=["csv"],
        help="CSV tối thiểu cần có cột: date, sales. Có thể thêm temperature, holiday."
    )


    try:
        df = load_data_from_upload(uploaded_file)
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return

    if uploaded_file is not None:
        st.success(f"Đã tải file: {uploaded_file.name}")
    else:
        st.warning("Chưa upload file. App đang dùng dữ liệu có sẵn hoặc dữ liệu mẫu.")

    using_uploaded_data = uploaded_file is not None

    st.subheader("2) Xem dữ liệu đầu vào")

    with st.expander("Xem toàn bộ dữ liệu"):
        st.dataframe(df, use_container_width=True)

    with st.expander("Thống kê mô tả dữ liệu"):
        st.dataframe(df.describe(include="all"), use_container_width=True)

    missing_required = []
    for col in ["date", "sales"]:
        if col not in df.columns:
            missing_required.append(col)

    if missing_required:
        st.error(f"Thiếu cột bắt buộc: {', '.join(missing_required)}")
    else:
        st.success("Dữ liệu đầu vào có đủ các cột bắt buộc: date, sales")

    with st.spinner("Đang xử lý dự báo..."):
        try:
            if uploaded_file is not None:
                eval_result = evaluate_saved_lstm_on_df(df)
                forecast_df = forecast_with_saved_lstm(df, days_ahead=forecast_days)

                y_true_lstm = eval_result["y_true"]
                y_pred_lstm = eval_result["y_pred"]
                lstm_mape = float(eval_result["mape"])
            else:
                # Dùng bộ LSTM đã train sẵn từ Colab
                forecast_df = pd.read_csv("forecast_result.csv")
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])

                lstm_test_df = pd.read_csv("lstm_test_result.csv")
                lstm_metrics_df = pd.read_csv("lstm_metrics.csv")

                y_true_lstm = lstm_test_df["y_true"].values
                y_pred_lstm = lstm_test_df["y_pred"].values
                lstm_mape = float(
                    lstm_metrics_df.loc[lstm_metrics_df["metric"] == "MAPE", "value"].iloc[0]
                )

            if scenario == "Tết":
                forecast_df["sales"] = (forecast_df["sales"] * 1.25).round().astype(int)

            elif scenario == "Mưa lũ HCMC":
                forecast_df["sales"] = (forecast_df["sales"] * 1.10).round().astype(int)
                lead_time_days = lead_time_days + 2

        except Exception as e:
            st.error(f"Lỗi xử lý dự báo: {e}")
            return
        if using_uploaded_data:
            st.success("App đang dùng mô hình LSTM train sẵn để dự báo trên chính file bạn upload.")
            st.info("Chế độ upload hiện load model LSTM đã train trước để tạo forecast cho dữ liệu mới.")
        else:
            st.success("App đang dùng đúng bộ dữ liệu gốc khớp với mô hình LSTM đã train.")
    inv = calculate_inventory_metrics(
        historical_sales=df["sales"],
        forecast_sales=forecast_df["sales"],
        lead_time_days=lead_time_days,
        z_value=z_value,
        order_cost=order_cost,
        holding_cost=holding_cost,
    ) 

    st.subheader("3) Kết quả AI Forecast & Inventory Optimization")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAPE", f"{lstm_mape:.2f}%")
    c2.metric("Nhu cầu TB/ngày", f"{inv.avg_daily_demand:.0f}")
    c3.metric("Safety Stock", f"{inv.safety_stock:.0f}")
    c4.metric("Reorder Point", f"{inv.reorder_point:.0f}")
    c5.metric("EOQ", f"{inv.eoq:.0f}")

    st.info("Hệ thống xuất ra nhu cầu dự báo, mức tồn kho an toàn, điểm đặt hàng lại và lượng đặt hàng kinh tế.")

    st.subheader("4) Cảnh báo tồn kho thông minh")
    current_stock = st.number_input(
        "Nhập mức tồn kho hiện tại",
        min_value=0,
        value=int(max(inv.reorder_point, inv.safety_stock) + 20),
        step=10
    )

    alert = evaluate_inventory_status(current_stock, inv, forecast_df)

    if alert["color"] == "error":
        st.error(f"**{alert['status']}** — {alert['message']}")
    elif alert["color"] == "warning":
        st.warning(f"**{alert['status']}** — {alert['message']}")
    else:
        st.success(f"**{alert['status']}** — {alert['message']}")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Mức ưu tiên", alert["priority"])
    a2.metric("Số ngày đủ hàng", f"{alert['days_of_cover']:.1f} ngày")
    a3.metric("Nhu cầu dự báo", f"{alert['next_7_days_demand']:.0f}")
    a4.metric("Chênh lệch vs ROP", f"{alert['stock_gap_vs_rop']:.0f}")

    if alert["suggested_order_qty"] > 0:
        st.info(f"Gợi ý nhập thêm khoảng **{alert['suggested_order_qty']}** đơn vị theo EOQ.")
    else:
        st.info("Hiện chưa cần nhập thêm theo tín hiệu tồn kho hiện tại.")

    st.subheader("5) Bảng dự báo")
    display_forecast = forecast_df.copy()
    display_forecast["date"] = display_forecast["date"].astype(str)
    display_forecast["sales"] = display_forecast["sales"].round(0).astype(int)
    st.dataframe(display_forecast, use_container_width=True)

    st.download_button(
        label="Tải kết quả dự báo xuống CSV",
        data=convert_df_to_csv(display_forecast),
        file_name="forecast_result.csv",
        mime="text/csv"
    )

    st.subheader("6) Biểu đồ")
    plot_actual_vs_forecast(df, forecast_df)
    plot_train_result(y_true_lstm, y_pred_lstm)

    st.subheader("7) Giải thích nhanh")
    st.markdown(
        f"""
**Mô hình AI sử dụng:**  
- Mặc định app dùng **TensorFlow LSTM (50 units)** đã train trên Google Colab.  
- Nếu người dùng upload file mới, app sẽ **train lại nhanh theo file upload** để tạo forecast phù hợp với dữ liệu đó.

**Input của mô hình:**  
- Lịch sử bán hàng theo ngày  
- Yếu tố mùa vụ / ngày trong tuần  
- Biến thời tiết và ngày lễ trong dữ liệu train

**Output của hệ thống:**  
- Nhu cầu dự báo trong các ngày tới  
- **Safety Stock** đề xuất  
- **Reorder Point** tối ưu  
- **EOQ** hỗ trợ quyết định nhập hàng

**Kịch bản mô phỏng hiện tại:**  
- **{scenario}**

**Độ chính xác hiện tại:**  
- **MAPE = {lstm_mape:.2f}%**
        """
    )


if __name__ == "__main__":
    main()
