from pathlib import Path
import os

import hopsworks
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from dotenv import load_dotenv


PROJECT_NAME = "ow_aqi_predictor"
FEATURE_STORE_NAME = "ow_aqi_predictor"
FEATURE_GROUP_NAME = "ow_aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_xgboost_model"
MODEL_DOWNLOAD_DIR = Path(".model_artifacts")

FEATURE_COLS = [
    "pm25",
    "pm10",
    "no2",
    "o3",
    "co",
    "so2",
    "nh3",
    "no",
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_24h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_std_6h",
    "aqi_rolling_max_24h",
    "aqi_change_rate",
    "pm25_to_pm10_ratio",
]
TARGET_COLS = ["aqi_next_24h", "aqi_next_48h", "aqi_next_72h"]
FORECAST_HORIZONS = ["24h", "48h", "72h"]
POLLUTANT_COLS = ["pm25", "pm10", "o3", "no2"]
POLLUTANT_LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "o3": "O3",
    "no2": "NO2",
}
AQI_COLORS = {
    "green": "#2ca25f",
    "yellow": "#fdd835",
    "orange": "#fb8c00",
    "red": "#e53935",
    "purple": "#8e24aa",
    "maroon": "#7f1d1d",
}


st.set_page_config(page_title="Rawalpindi AQI Forecast", layout="wide")


@st.cache_resource(show_spinner=False)
def connect_to_hopsworks(api_key: str):
    project = hopsworks.login(
        project=PROJECT_NAME,
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=api_key,
    )
    return project, project.get_feature_store(name=FEATURE_STORE_NAME), project.get_model_registry()


def find_artifact_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


@st.cache_resource(show_spinner=False)
def load_production_artifacts(api_key: str):
    project, _, model_registry = connect_to_hopsworks(api_key)
    registered_model = model_registry.get_best_model(MODEL_NAME, metric="rmse", direction="min")

    if registered_model is None:
        registered_model = model_registry.get_model(MODEL_NAME)

    if registered_model is None:
        st.error(f"Model Registry does not contain a model named {MODEL_NAME}.")
        st.stop()

    model_path = Path(registered_model.download(local_path=str(MODEL_DOWNLOAD_DIR)))
    model_file = find_artifact_file(model_path, "xgb_model.pkl") or Path("xgb_model.pkl")
    scaler_file = find_artifact_file(model_path, "scaler.pkl") or Path("scaler.pkl")

    missing_files = [str(path) for path in [model_file, scaler_file] if not path.exists()]
    if missing_files:
        st.error("Missing required model artifact(s): " + ", ".join(missing_files))
        st.stop()

    if scaler_file == Path("scaler.pkl"):
        st.warning("scaler.pkl was not found in the downloaded Model Registry artifact. Using local scaler.pkl.")

    return joblib.load(model_file), joblib.load(scaler_file), str(model_path)


@st.cache_data(show_spinner=False, ttl=600)
def read_feature_data(api_key: str):
    _, feature_store, _ = connect_to_hopsworks(api_key)
    feature_group = feature_store.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    data = feature_group.read()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data.sort_values("timestamp")


def latest_valid_features(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    missing_cols = [col for col in ["timestamp", "aqi", *FEATURE_COLS] if col not in data.columns]
    if missing_cols:
        st.error("Feature group is missing required column(s): " + ", ".join(missing_cols))
        st.stop()

    clean_data = data.ffill(limit=3).dropna(subset=FEATURE_COLS)
    if clean_data.empty:
        st.error("No row has all required model features after forward filling.")
        st.stop()

    latest_row = clean_data.iloc[-1]
    latest_raw_features = latest_row[FEATURE_COLS].to_frame().T
    return latest_row, latest_raw_features


def normalize_forecast(prediction) -> np.ndarray:
    forecast = np.asarray(prediction).reshape(-1)
    if forecast.size < 3:
        st.error("Model prediction did not return 24h, 48h, and 72h forecasts.")
        st.stop()
    return forecast[:3]


def aqi_band(value: float) -> tuple[str, str]:
    if value <= 1:
        return "Good", AQI_COLORS["green"]
    if value <= 2:
        return "Fair", AQI_COLORS["yellow"]
    if value <= 3:
        return "Moderate", AQI_COLORS["orange"]
    if value <= 4:
        return "Unhealthy", AQI_COLORS["red"]
    if value <= 5:
        return "Very Unhealthy", AQI_COLORS["purple"]
    return "Hazardous", AQI_COLORS["maroon"]


def render_aqi_gauge(current_aqi: float):
    band_name, band_color = aqi_band(current_aqi)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            number={"suffix": f" {band_name}", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 6], "tickvals": [1, 2, 3, 4, 5, 6]},
                "bar": {"color": band_color},
                "steps": [
                    {"range": [0, 1], "color": AQI_COLORS["green"]},
                    {"range": [1, 2], "color": AQI_COLORS["yellow"]},
                    {"range": [2, 3], "color": AQI_COLORS["orange"]},
                    {"range": [3, 4], "color": AQI_COLORS["red"]},
                    {"range": [4, 5], "color": AQI_COLORS["purple"]},
                    {"range": [5, 6], "color": AQI_COLORS["maroon"]},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75,
                    "value": 4,
                },
            },
        )
    )
    fig.update_layout(height=320, margin={"l": 20, "r": 20, "t": 20, "b": 10})
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_chart(forecast: np.ndarray):
    forecast_df = pd.DataFrame(
        {
            "horizon": FORECAST_HORIZONS,
            "forecast_aqi": forecast,
        }
    )
    fig = px.line(
        forecast_df,
        x="horizon",
        y="forecast_aqi",
        markers=True,
        labels={"horizon": "Forecast horizon", "forecast_aqi": "Forecast AQI"},
    )
    fig.update_traces(line={"width": 3, "color": "#2563eb"}, marker={"size": 10})
    fig.update_layout(yaxis_range=[0, max(6, float(np.nanmax(forecast)) + 0.5)])
    st.plotly_chart(fig, use_container_width=True)


def render_pollutant_breakdown(latest_row: pd.Series):
    pollutant_values = latest_row[POLLUTANT_COLS].astype(float).clip(lower=0)
    total = pollutant_values.sum()
    if total == 0:
        st.info("Pollutant contribution chart is unavailable because all selected pollutant values are zero.")
        return

    contribution_df = pd.DataFrame(
        {
            "pollutant": [POLLUTANT_LABELS[col] for col in POLLUTANT_COLS],
            "contribution_percent": (pollutant_values / total * 100).values,
        }
    )
    fig = px.bar(
        contribution_df,
        x="pollutant",
        y="contribution_percent",
        labels={"pollutant": "Pollutant", "contribution_percent": "Contribution (%)"},
        color="pollutant",
        color_discrete_map={
            "PM2.5": "#7c3aed",
            "PM10": "#2563eb",
            "O3": "#f59e0b",
            "NO2": "#dc2626",
        },
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


def shap_values_for_first_horizon(explainer, features: pd.DataFrame):
    explanation = explainer(features)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        values = values[0, :, 0]
        base_value = base_values[0, 0] if base_values.ndim > 1 else base_values[0]
    elif values.ndim == 2:
        values = values[0]
        base_value = base_values[0] if base_values.ndim else base_values.item()
    else:
        values = values.reshape(-1)
        base_value = base_values.reshape(-1)[0]

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=features.iloc[0].to_numpy(),
        feature_names=FEATURE_COLS,
    )


def render_shap_waterfall(model, latest_features: pd.DataFrame):
    try:
        explainer = shap.Explainer(model, latest_features)
        explanation = shap_values_for_first_horizon(explainer, latest_features)
        plt.figure()
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    except Exception as exc:
        st.warning(f"SHAP waterfall chart could not be generated: {exc}")


load_dotenv()
hw_key = os.getenv("HOPSWORKS_OW_API_KEY")

if not hw_key:
    st.error("Missing HOPSWORKS_OW_API_KEY in .env")
    st.stop()

st.title("Rawalpindi AQI Forecast Dashboard")

with st.spinner("Loading production model, scaler, and latest features..."):
    model, scaler, artifact_path = load_production_artifacts(hw_key)
    df = read_feature_data(hw_key)
    latest_row, latest_raw_features = latest_valid_features(df)
    latest_features = pd.DataFrame(
        scaler.transform(latest_raw_features),
        columns=FEATURE_COLS,
        index=latest_raw_features.index,
    )
    forecast_values = normalize_forecast(model.predict(latest_features))

current_aqi = float(latest_row["aqi"])

if np.nanmax(forecast_values) > 4:
    st.error("Alert: at least one forecasted AQI value exceeds 4 (Unhealthy).")

top_metrics = st.columns(4)
top_metrics[0].metric("Current AQI", f"{current_aqi:.1f}", aqi_band(current_aqi)[0])
for column, horizon, value in zip(top_metrics[1:], FORECAST_HORIZONS, forecast_values):
    column.metric(f"{horizon} forecast", f"{value:.2f}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Current AQI")
    render_aqi_gauge(current_aqi)

with right:
    st.subheader("3-day forecast")
    render_forecast_chart(forecast_values)

lower_left, lower_right = st.columns([1, 1])
with lower_left:
    st.subheader("Pollutant contribution")
    render_pollutant_breakdown(latest_row)

with lower_right:
    st.subheader("SHAP feature importance")
    render_shap_waterfall(model, latest_features)

with st.expander("Latest feature row"):
    st.caption(f"Model artifacts loaded from: {artifact_path}")
    st.dataframe(df.tail(10), use_container_width=True)
