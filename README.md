#  Rawalpindi AQI Forecast

A fully serverless, end-to-end **Air Quality Index (AQI) prediction system** for Rawalpindi, Pakistan. The system automatically collects live pollution and weather data every hour, trains an XGBoost model daily, and serves 3-day AQI forecasts through an interactive Streamlit dashboard — all without managing any servers.

---

##  Dashboard Preview

> Live dashboard displays the current AQI gauge, 3-day forecast chart, pollutant breakdown, and SHAP feature importance — all refreshed automatically from the Feature Store.

---

## 🏗️ Architecture

```
OpenWeather / AQI API
        │
        ▼ (every hour via GitHub Actions)
┌─────────────────────┐
│   Feature Pipeline  │  → computes 22 engineered features
└────────┬────────────┘
         │ writes
         ▼
┌─────────────────────┐
│  Hopsworks Feature  │  ← single source of truth
│  Store & Model Reg. │
└────────┬────────────┘
         │ reads (daily via GitHub Actions)
         ▼
┌─────────────────────┐
│  Training Pipeline  │  → trains XGBoost, evaluates RMSE/MAE/R²
└────────┬────────────┘
         │ saves model
         ▼
┌─────────────────────┐
│   Streamlit App     │  → loads model + features → shows forecast
└─────────────────────┘
```

---

##  Features

- **3-day AQI forecast** — predicts air quality at 24h, 48h, and 72h horizons
- **Live data ingestion** — pulls pollutant and weather data from OpenWeather API every hour
- **22 engineered features** — lag features (1h, 3h, 6h, 24h), rolling statistics, time-based features, and derived ratios
- **XGBoost model** — trained and evaluated daily with RMSE, MAE, and R² metrics
- **SHAP explainability** — waterfall chart showing which features drove each prediction
- **Pollutant breakdown** — bar chart showing PM2.5, PM10, O3, NO2 contributions
- **Hazard alerts** — automatic banner when any forecasted AQI exceeds the Unhealthy threshold
- **Fully automated** — GitHub Actions runs the feature pipeline hourly and training pipeline daily
- **Serverless** — no servers to manage; Hopsworks free tier + Streamlit Cloud + GitHub Actions

---

##  Project Structure

```
OpenWeather/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml   # Runs every hour
│       └── training_pipeline.yml  # Runs every day at 02:00 UTC
├── feature_pipeline/
│   ├── feature_pipeline.py        # Fetches data, engineers features, writes to Hopsworks
├── training_pipeline/
│   └── train.py                   # Trains XGBoost, evaluates, saves to Model Registry
├── streamlit_app.py               # Interactive forecast dashboard
├── openWeather.ipynb              # Exploration and EDA notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Tech Stack

| Layer | Tool |
|---|---|
| Data source | OpenWeather Air Pollution API |
| Feature store | Hopsworks (free tier) |
| Model registry | Hopsworks Model Registry |
| ML model | XGBoost (multi-output: 24h / 48h / 72h) |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| Automation | GitHub Actions |
| Language | Python 3.10+ |

---

## ⚙️ Engineered Features

The model uses 22 features computed from raw API data:

**Pollutants:** `pm25`, `pm10`, `no2`, `o3`, `co`, `so2`, `nh3`, `no`

**Time-based:** `hour_of_day`, `day_of_week`, `day_of_month`, `month`, `is_weekend`

**Lag features:** `aqi_lag_1h`, `aqi_lag_3h`, `aqi_lag_6h`, `aqi_lag_24h`

**Rolling statistics:** `aqi_rolling_mean_6h`, `aqi_rolling_std_6h`, `aqi_rolling_max_24h`

**Derived:** `aqi_change_rate`, `pm25_to_pm10_ratio`

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/irfanthetech/OpenWeather.git
cd OpenWeather
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
OPENWEATHER_API_KEY=your_openweather_api_key_here
HOPSWORKS_OW_API_KEY=your_hopsworks_api_key_here
```

- Get your OpenWeather API key at [openweathermap.org/api](https://openweathermap.org/api) (free tier)
- Get your Hopsworks API key at [app.hopsworks.ai](https://app.hopsworks.ai) → Project Settings → API Keys

### 4. Run the feature pipeline

```bash
python feature_pipeline/feature_pipeline.py
```


### 5. Train the model

```bash
python training_pipeline/train.py
```

### 6. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

---

##  Automated Pipelines (GitHub Actions)

The project uses two GitHub Actions workflows:

| Workflow | Schedule | What it does |
|---|---|---|
| `feature_pipeline.yml` | Every hour (`0 * * * *`) | Fetches live data, engineers features, writes to Hopsworks |
| `training_pipeline.yml` | Daily at 02:00 UTC (`0 2 * * *`) | Retrains XGBoost on latest data, saves best model to registry |

### Setting up secrets

Go to your GitHub repo → **Settings → Secrets and variables → Actions** and add:

- `OPENWEATHER_API_KEY`
- `HOPSWORKS_OW_API_KEY`

---

## 📊 AQI Scale Reference

| AQI Range | Category | Health Advisory |
|---|---|---|
| 0 – 1 | 🟢 Good | No health concern |
| 1 – 2 | 🟡 Moderate | Acceptable for most people |
| 2 – 3 | 🟠 Unhealthy for Sensitive Groups | Sensitive groups at risk |
| 3 – 4 | 🔴 Unhealthy | Everyone may experience effects |
| 4 – 5 | 🟣 Very Unhealthy | Health alert for all |
| 5 | 🔴 Hazardous | Emergency conditions |

---

## 📈 Model Performance

The XGBoost model is evaluated on a time-based holdout set (last 20% of historical data) across all three forecast horizons:



> Metrics are updated after each daily training run. Fill these in from your latest training output.

---

## 📁 Hopsworks Configuration

| Setting | Value |
|---|---|
| Project name | `ow_aqi_predictor` |
| Feature group | `ow_aqi_features` (version 1) |
| Model name | `aqi_xgboost_model` |
| Host | `eu-west.cloud.hopsworks.ai` |

---

##  About

Built by **Irfan** as part of the Pearls AQI Predictor project — an end-to-end serverless ML pipeline for air quality forecasting using real-time data from Rawalpindi, Pakistan.
