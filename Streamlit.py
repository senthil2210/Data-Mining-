%%writefile streamlit_app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(
    page_title="EV Charging Forecasting & Pricing",
    layout="wide"
)

CALIF_PATH   = "Charging station_A_Calif.csv"
PATTERNS_PATH = "ev_charging_patterns.csv"

TARGET_COL   = "EV Charging Demand (kW)"
CAPACITY_COL = "Charging Station Capacity (kW)"
PRICE_COL    = "Electricity Price ($/kWh)"


@st.cache_data
def load_calif():
    df = pd.read_csv(CALIF_PATH)
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL]).copy()

    df[CAPACITY_COL] = pd.to_numeric(df[CAPACITY_COL], errors="coerce")
    df[PRICE_COL]    = pd.to_numeric(df[PRICE_COL], errors="coerce")

    df[CAPACITY_COL] = df[CAPACITY_COL].fillna(df[CAPACITY_COL].median())
    df[PRICE_COL]    = df[PRICE_COL].fillna(df[PRICE_COL].median())

    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"]     = df["timestamp"].dt.month
    df["date"]      = df["timestamp"].dt.date

    return df


@st.cache_data
def load_sessions():
    df = pd.read_csv(PATTERNS_PATH)
    for col in ["Charging Start Time", "Charging End Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_resource
def train_xgb(df_calif):
    feature_cols = [
        c for c in df_calif.columns
        if c not in ["timestamp", "Date", "Time", "date", TARGET_COL]
        and df_calif[c].dtype != "O"
    ]
    X = df_calif[feature_cols]
    y = df_calif[TARGET_COL]

    split_idx = int(len(df_calif) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "feature_importance": fi,
        "split_idx": split_idx,
    }


@st.cache_resource
def train_lstm(df_calif, window=24):
    values = df_calif[[TARGET_COL]].values.astype(float)

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    def create_sequences(data, win):
        Xs, ys = [], []
        for i in range(len(data) - win):
            Xs.append(data[i:i+win])
            ys.append(data[i+win])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(values_scaled, window)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    X_train = X_train.reshape((-1, window, 1))
    X_test  = X_test.reshape((-1, window, 1))

    model = Sequential([
        LSTM(64, input_shape=(window, 1), return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    y_pred_scaled = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    mae  = mean_absolute_error(y_test_inv, y_pred)
    r2   = r2_score(y_test_inv, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "window": window,
        "X_test": X_test,
        "y_test_inv": y_test_inv,
        "y_pred": y_pred,
        "history": history.history,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


@st.cache_resource
def train_multi_horizon(df_calif, horizons=None, n_lags=24):
    if horizons is None:
        horizons = [1, 6, 12, 24]

    df_mh = df_calif[["timestamp", TARGET_COL]].copy()

    for lag in range(1, n_lags + 1):
        df_mh[f"lag_{lag}"] = df_mh[TARGET_COL].shift(lag)

    for h in horizons:
        df_mh[f"ahead_{h}"] = df_mh[TARGET_COL].shift(-h)

    df_mh = df_mh.dropna().reset_index(drop=True)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    target_cols  = [f"ahead_{h}" for h in horizons]

    X = df_mh[feature_cols]
    Y = df_mh[target_cols]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

    base_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    mh_model = MultiOutputRegressor(base_model)
    mh_model.fit(X_train, Y_train)

    Y_pred = mh_model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=target_cols, index=Y_test.index)

    metrics = []
    for h in horizons:
        col = f"ahead_{h}"
        true = Y_test[col].values
        pred = Y_pred_df[col].values
        mae  = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        metrics.append({"Horizon (h)": h, "MAE": mae, "RMSE": rmse})

    metrics_df = pd.DataFrame(metrics)

    return {
        "model": mh_model,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred_df": Y_pred_df,
        "metrics_df": metrics_df,
        "horizons": horizons,
    }


@st.cache_data
def build_station_profile(df_sessions):
    group_cols = ["Charging Station ID", "Charging Station Location"]
    for g in group_cols:
        if g not in df_sessions.columns:
            raise ValueError(f"Column '{g}' not found in session data.")

    station_profile = df_sessions.groupby(group_cols, as_index=False).agg(
        sessions=("User ID", "count"),
        avg_energy=("Energy Consumed (kWh)", "mean"),
        avg_duration=("Charging Duration (hours)", "mean"),
        avg_rate=("Charging Rate (kW)", "mean"),
        avg_cost=("Charging Cost (USD)", "mean")
    )

    station_profile["load_score"] = (
        station_profile["sessions"] *
        station_profile["avg_duration"].fillna(0) *
        station_profile["avg_rate"].fillna(0)
    )

    return station_profile


def recommend_stations(station_profile, location: str, top_n: int = 5, min_sessions: int = 10):
    df_loc = station_profile[station_profile["Charging Station Location"] == location].copy()
    if df_loc.empty:
        return pd.DataFrame()
    df_loc = df_loc[df_loc["sessions"] >= min_sessions]
    if df_loc.empty:
        df_loc = station_profile[station_profile["Charging Station Location"] == location].copy()
    return df_loc.sort_values("load_score").head(top_n)


def dynamic_price(base_price, utilization, alpha=0.8, floor_mult=0.7, cap_mult=2.0, threshold=0.7):
    util = np.clip(utilization, 0, None)
    surge_factor = 1 + alpha * (util - threshold)
    surge_factor = np.where(util < threshold, 1.0, surge_factor)
    surge_factor = np.clip(surge_factor, floor_mult, cap_mult)
    return base_price * surge_factor


df_calif = load_calif()
df_sessions = load_sessions()
station_profile = build_station_profile(df_sessions)

st.title("⚡ EV Charging Forecasting, Recommendations & Dynamic Pricing")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "California EDA + XGBoost",
    "LSTM Forecasting",
    "Multi-Horizon Forecasting",
    "Station Recommendations",
    "Dynamic Pricing"
])

with tab1:
    st.subheader("California Station – Demand EDA & XGBoost Forecast")

    st.dataframe(df_calif.head())

    st.line_chart(df_calif.set_index("timestamp")[TARGET_COL])

    daily_demand = df_calif.groupby("date")[TARGET_COL].mean()
    st.line_chart(daily_demand)

    xgb_art = train_xgb(df_calif)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{xgb_art['rmse']:.4f}")
    c2.metric("MAE", f"{xgb_art['mae']:.4f}")
    c3.metric("R²", f"{xgb_art['r2']:.4f}")

    st.write("Top 10 feature importances:")
    st.dataframe(xgb_art["feature_importance"].head(10))

    fig, ax = plt.subplots(figsize=(10, 4))
    ts_test = df_calif["timestamp"].iloc[xgb_art["split_idx"]:]
    ax.plot(ts_test, xgb_art["y_test"].values, label="Actual")
    ax.plot(ts_test, xgb_art["y_pred"], label="Predicted", linestyle="--")
    ax.set_xticklabels(ts_test.dt.strftime("%Y-%m-%d %H:%M"), rotation=45)
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("LSTM Single-Step Forecasting")

    lstm_art = train_lstm(df_calif, window=24)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{lstm_art['rmse']:.4f}")
    c2.metric("MAE", f"{lstm_art['mae']:.4f}")
    c3.metric("R²", f"{lstm_art['r2']:.4f}")

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(lstm_art["history"]["loss"], label="train_loss")
    ax_loss.plot(lstm_art["history"]["val_loss"], label="val_loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

    fig_lstm, ax_lstm = plt.subplots()
    y_true = lstm_art["y_test_inv"][-200:]
    y_pred = lstm_art["y_pred"][-200:]
    ax_lstm.plot(y_true, label="Actual")
    ax_lstm.plot(y_pred, label="Predicted", linestyle="--")
    ax_lstm.set_ylabel(TARGET_COL)
    ax_lstm.legend()
    st.pyplot(fig_lstm)

with tab3:
    st.subheader("Multi-Horizon Forecasting (1h, 6h, 12h, 24h)")

    mh_art = train_multi_horizon(df_calif)
    st.dataframe(mh_art["metrics_df"])

    horizon = st.selectbox("Horizon (hours):", mh_art["horizons"], index=mh_art["horizons"].index(24))
    col_plot = f"ahead_{horizon}"

    plot_len = st.slider(
        "Number of last test samples to visualize:",
        min_value=50,
        max_value=len(mh_art["Y_test"]),
        value=min(200, len(mh_art["Y_test"])),
        step=25
    )

    y_true = mh_art["Y_test"][col_plot].tail(plot_len).reset_index(drop=True)
    y_pred = mh_art["Y_pred_df"][col_plot].tail(plot_len).reset_index(drop=True)

    fig_mh, ax_mh = plt.subplots(figsize=(10, 4))
    ax_mh.plot(y_true, label="Actual")
    ax_mh.plot(y_pred, label="Predicted", linestyle="--")
    ax_mh.set_xlabel("Index (within test)")
    ax_mh.set_ylabel(TARGET_COL)
    ax_mh.legend()
    st.pyplot(fig_mh)

with tab4:
    st.subheader("Station Recommendations")

    st.dataframe(station_profile.head())

    locations = sorted(station_profile["Charging Station Location"].unique())
    loc = st.selectbox("Location:", locations)

    min_sessions = st.slider("Minimum sessions per station:", 1, 50, 10, 1)
    top_n = st.slider("Top N stations:", 1, 10, 5, 1)

    recs = recommend_stations(station_profile, loc, top_n=top_n, min_sessions=min_sessions)

    if recs.empty:
        st.warning("No stations match filters for this location.")
    else:
        st.dataframe(recs)

with tab5:
    st.subheader("Dynamic Pricing Based on Utilization")

    df_price = df_calif.copy()
    df_price["utilization"] = df_price[TARGET_COL] / df_price[CAPACITY_COL].replace(0, np.nan)

    alpha = st.slider("Surge sensitivity (alpha):", 0.1, 2.0, 0.8, 0.1)
    floor_mult = st.slider("Minimum price multiplier:", 0.5, 1.0, 0.7, 0.05)
    cap_mult = st.slider("Maximum price multiplier:", 1.0, 3.0, 2.0, 0.1)
    threshold = st.slider("Utilization threshold for surge:", 0.4, 1.0, 0.7, 0.05)

    df_price["dynamic_price"] = dynamic_price(
        df_price[PRICE_COL].values,
        df_price["utilization"].fillna(0).values,
        alpha=alpha,
        floor_mult=floor_mult,
        cap_mult=cap_mult,
        threshold=threshold
    )

    st.dataframe(
        df_price[[
            "timestamp", TARGET_COL, CAPACITY_COL, PRICE_COL,
            "utilization", "dynamic_price"
        ]].head(20)
    )

    window = st.slider("Last N points to plot:", 100, len(df_price), min(300, len(df_price)), 50)
    sub = df_price.tail(window).set_index("timestamp")

    st.line_chart(sub[[TARGET_COL, CAPACITY_COL]])
    st.line_chart(sub[[PRICE_COL, "dynamic_price"]])
