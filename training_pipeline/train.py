from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import hopsworks
import joblib 
import os 
import hsml
from hsml.model_schema import ModelSchema
from dotenv import load_dotenv
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
load_dotenv()
hw_key = os.getenv("HOPSWORKS_API_KEY")
if not hw_key:
    raise RuntimeError("Missing HOPSWORKS_OW_API_KEY in .env")

project = hopsworks.login(
    project="ow_aqi_predictor",
    host="eu-west.cloud.hopsworks.ai",  
    api_key_value=hw_key
)
mr = project.get_model_registry()
fs = project.get_feature_store(name='ow_aqi_predictor')
mr = project.get_model_registry()
fg = fs.get_feature_group('ow_aqi_features', version=1)
df = fg.read()
df = df.ffill(limit=3)

# Drop rows with >30% missing values
row_missing_fraction = df.isnull().mean(axis=1)
df = df[row_missing_fraction <= 0.3].reset_index(drop=True)

# Define features and targets
feature_cols = [
    "pm25","pm10","no2","o3","co","so2","nh3","no",
    "hour_of_day","day_of_week","day_of_month","month","is_weekend",
    "aqi_lag_1h","aqi_lag_3h","aqi_lag_6h","aqi_lag_24h",
    "aqi_rolling_mean_6h","aqi_rolling_std_6h","aqi_rolling_max_24h",
    "aqi_change_rate","pm25_to_pm10_ratio"
]
target_cols = ["aqi_next_24h","aqi_next_48h","aqi_next_72h"]

# Drop rows with NaNs in features or targets
df_clean = df.dropna(subset=feature_cols + target_cols)

# Redefine X and y
X = df_clean[feature_cols].values
y = df_clean[target_cols].values

# Time-based split
split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
# --- Scaling ---
# StandardScaler: good for normally distributed features
# RobustScaler: better if AQI/pollutants have outliers
scaler = RobustScaler()   # or StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#print("Train shape:", X_train_scaled.shape, "Test shape:", X_test_scaled.shape)
joblib.dump(scaler, "scaler.pkl")

xgb = XGBRegressor(
    n_estimators=300,     # number of boosting rounds
    learning_rate=0.05,   # step size shrinkage
    max_depth=6,          # depth of trees
    subsample=0.8,        # row sampling
    colsample_bytree=0.8, # feature sampling
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# --- Predictions ---
y_pred = xgb.predict(X_test)

# --- Evaluation ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#print("XGBoost Results")
#print("RMSE:", rmse)
#print("R²:", r2)



# Define input schema (features with types)
input_schema = hsml.schema.Schema([
    {"name": "pm25", "type": "float"},
    {"name": "pm10", "type": "float"},
    {"name": "no2", "type": "float"},
    {"name": "o3", "type": "float"},
    {"name": "co", "type": "float"},
    {"name": "so2", "type": "float"},
    {"name": "nh3", "type": "float"},
    {"name": "no", "type": "float"},
    {"name": "hour_of_day", "type": "int"},
    {"name": "day_of_week", "type": "int"},
    {"name": "day_of_month", "type": "int"},
    {"name": "month", "type": "int"},
    {"name": "is_weekend", "type": "int"},
    {"name": "aqi_lag_1h", "type": "float"},
    {"name": "aqi_lag_3h", "type": "float"},
    {"name": "aqi_lag_6h", "type": "float"},
    {"name": "aqi_lag_24h", "type": "float"},
    {"name": "aqi_rolling_mean_6h", "type": "float"},
    {"name": "aqi_rolling_std_6h", "type": "float"},
    {"name": "aqi_rolling_max_24h", "type": "float"},
    {"name": "aqi_change_rate", "type": "float"},
    {"name": "pm25_to_pm10_ratio", "type": "float"}
])

# Define output schema (targets with types)
output_schema = hsml.schema.Schema([
    {"name": "aqi_next_24h", "type": "float"},
    {"name": "aqi_next_48h", "type": "float"},
    {"name": "aqi_next_72h", "type": "float"}
])

# Bundle into model schema
model_schema = ModelSchema(input_schema, output_schema)
# Assume you already trained XGBoost and have best_rmse
best_rmse = 0.3244120840471264  # replace with your actual RMSE

# Register the model
model = mr.python.create_model(
    name="aqi_xgboost_model",
    metrics={"rmse": best_rmse},
    model_schema=model_schema,
    description="XGBoost model predicting AQI for next 24h, 48h, 72h"
)
# Save trained XGBoost model
joblib.dump(xgb, "xgb_model.pkl")

# Save the trained model object
model.save("xgb_model.pkl")
