import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import joblib

CSV_PATH = 'crop_yield.csv'
TARGET = 'Yield'   

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("shape:", df.shape)
print(df.columns.tolist())


if 'Production' in df.columns and 'Area' in df.columns and TARGET == 'Yield' and 'Yield' not in df.columns:
    df['Yield'] = df['Production'] / df['Area']

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

candidate_features = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
features = [c for c in candidate_features if c in df.columns]
if not features:
    features = [c for c in df.columns if c != TARGET]

X = df[features].copy()
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')


model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
print("Running CV (this may take a bit)...")
cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
print("CV RMSE:", -cv_res['test_rmse'].mean())
print("CV MAE: ", -cv_res['test_mae'].mean())
print("CV R2:  ", cv_res['test_r2'].mean())

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print("TEST  RMSE:", rmse)
print("TEST  MAE: ", mae)
print("TEST  R2:  ", r2)

import joblib, json
from datetime import datetime
from pathlib import Path
import numpy as np

out_dir = Path('./outputs_2')  
out_dir.mkdir(parents=True, exist_ok=True)

ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
pipeline_filename = out_dir / f'crop_yield_pipeline_{ts}.joblib'

joblib.dump(pipeline, pipeline_filename, compress=3)
print("Saved pipeline to:", pipeline_filename)

estimator_name = None
for candidate in ('estimator', 'model'):
    if candidate in pipeline.named_steps:
        estimator_name = candidate
        break
if estimator_name is None:
    estimator_name = list(pipeline.named_steps.keys())[-1]

estimator_obj = pipeline.named_steps[estimator_name]
print(f"Detected estimator step name: '{estimator_name}' -> {type(estimator_obj).__name__}")
try:
    feature_list = list(X_train.columns)
except Exception:
    feature_list = []

metadata = {
    'created_at': ts,
    'model_file': pipeline_filename.name,
    'model_type': type(estimator_obj).__name__,
    'estimator_step_name': estimator_name,
    'features': feature_list,
    'target': 'Yield'
}
with open(out_dir / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Saved metadata to model_metadata.json")

train_preds = pipeline.predict(X_train)
residuals = (y_train - train_preds)
resid_std = float(np.std(residuals))
resid_q = list(np.percentile(residuals, [2.5, 50.0, 97.5]))
resid_info = {'resid_std': resid_std, 'resid_q': resid_q}
with open(out_dir / 'residual_stats.json', 'w') as f:
    json.dump(resid_info, f, indent=2)
print("Saved residual_stats.json (used for prediction intervals)")

latest_path = out_dir / 'crop_yield_pipeline_latest.joblib'
joblib.dump(pipeline, latest_path, compress=3)
print("Also saved 'latest' pipeline to:", latest_path)
