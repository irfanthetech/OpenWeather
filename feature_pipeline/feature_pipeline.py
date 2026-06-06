import requests as req
import os 
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from datetime import datetime, UTC, timedelta
load_dotenv()
ow_key = os.getenv("OPENWEATHER_API_KEY")
hw_key = os.getenv("HOPSWORKS_API_KEY")
lat, lon = 33.63, 73.04  
ow_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={ow_key}"

ow_response = req.get(ow_url).json()
#print(ow_response )
data = ow_response['list'][0]
aqi = data['main']['aqi']
components = data['components']
date = ow_response['list'][0]['dt']

row = {
    "timestamp": datetime.fromtimestamp(date, UTC),
    "city": "Rawalpindi",
    "aqi": aqi,
    "pm25": components['pm2_5'],
    "pm10": components['pm10'],
    "no2": components['no2'],
    "o3": components['o3'],
    "co": components['co'],
    "so2": components['so2'],
    "nh3": components['nh3'],
    "no": components['no']
}

df = pd.DataFrame([row])
df.head()
pollutant_columns = ["pm25", "pm10", "no2", "o3", "co", "so2", "nh3", "no"]
df[pollutant_columns] = df[pollutant_columns].astype("float64")

df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek   # Monday=0
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
#df.head()
df['aqi_lag_1h'] = df['aqi'].shift(1)
df['aqi_lag_3h'] = df['aqi'].shift(3)
df['aqi_lag_6h'] = df['aqi'].shift(6)
df['aqi_lag_24h'] = df['aqi'].shift(24)
#df.head()
df['aqi_rolling_mean_6h'] = df['aqi'].rolling(window=6).mean()
df['aqi_rolling_std_6h'] = df['aqi'].rolling(window=6).std()
df['aqi_rolling_max_24h'] = df['aqi'].rolling(window=24).max()
#df.head()
# AQI change rate (difference per hour)
df['aqi_change_rate'] = df['aqi'].diff() / 1  # since hourly

# PM2.5 to PM10 ratio
df['pm25_to_pm10_ratio'] = df['pm25'] / df['pm10']
df['aqi_next_24h'] = df['aqi'].shift(-24)
df['aqi_next_48h'] = df['aqi'].shift(-48)
df['aqi_next_72h'] = df['aqi'].shift(-72)
#df = df.dropna()

#df.head()
#df.columns.tolist()
# This temporarily displays all columns just for this block
#with pd.option_context('display.max_columns', None, 'display.width', 1000):
  #  print(df)



# Connect to Hopsworks


project = hopsworks.login(
    project="ow_aqi_predictor",   # Replace with your project name
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=hw_key   # Get from Hopsworks UI > Account Settings > API Keys
)

# Access the feature store
fs = project.get_feature_store()

# Create or get your AQI feature group
fg = fs.get_or_create_feature_group(
    name="ow_aqi_features",
    version=1,
    primary_key=["city", "timestamp"],
    event_time="timestamp"
)



def compute_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    pollutant_columns = ["pm25", "pm10", "no2", "o3", "co", "so2", "nh3", "no"]
    df[pollutant_columns] = df[pollutant_columns].astype("float64")

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)
    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    df["aqi_rolling_mean_6h"] = df["aqi"].rolling(window=6).mean()
    df["aqi_rolling_std_6h"] = df["aqi"].rolling(window=6).std()
    df["aqi_rolling_max_24h"] = df["aqi"].rolling(window=24).max()

    df["aqi_change_rate"] = df["aqi"].diff()
    df["pm25_to_pm10_ratio"] = df["pm25"] / df["pm10"]

    df["aqi_next_24h"] = df["aqi"].shift(-24)
    df["aqi_next_48h"] = df["aqi"].shift(-48)
    df["aqi_next_72h"] = df["aqi"].shift(-72)

    return df


def fetch_air_pollution_history_for_day(day, ow_key, lat, lon):
    start_dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
    end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)

    start = int(start_dt.timestamp())
    end = int(end_dt.timestamp())

    ow_history = (
        "https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start}&end={end}&appid={ow_key}"
    )

    ow_response = req.get(ow_history).json()
    rows = []

    for item in ow_response.get("list", []):
        components = item["components"]

        rows.append({
            "timestamp": datetime.fromtimestamp(item["dt"], UTC),
            "city": "Rawalpindi",
            "aqi": item["main"]["aqi"],
            "pm25": components["pm2_5"],
            "pm10": components["pm10"],
            "no2": components["no2"],
            "o3": components["o3"],
            "co": components["co"],
            "so2": components["so2"],
            "nh3": components["nh3"],
            "no": components["no"],
        })

    return rows


ow_key = os.getenv("openweather_key")
lat, lon = 33.63, 73.04

start_date = datetime.strptime("01-05-2026", "%d-%m-%Y").date()
end_date = (datetime.now(UTC) - timedelta(days=3)).date()

all_rows = []
current_date = start_date

while current_date <= end_date:
    rows = fetch_air_pollution_history_for_day(
        day=current_date,
        ow_key=ow_key,
        lat=lat,
        lon=lon
    )

    all_rows.extend(rows)
    current_date += timedelta(days=1)

historical_df = pd.DataFrame(all_rows)

if not historical_df.empty:
    historical_df = compute_features(historical_df)
    fg.insert(historical_df)
