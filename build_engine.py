import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

print("Loading dataset India_Crop_Production.csv...")
df = pd.read_csv("India_Crop_Production.csv")

# Clean dataset
df = df.dropna()
df = df[(df['Area'] > 0) & (df['Production'] >= 0)]

# Standardize string columns (strip whitespace)
for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
    df[col] = df[col].astype(str).str.strip()

# Calculate Yield = Production / Area (Tons per Hectare)
df['Yield'] = df['Production'] / df['Area']

# Remove extreme outliers in Yield (e.g. > 99.9th percentile per crop or global > 500)
q99 = df['Yield'].quantile(0.999)
df = df[df['Yield'] <= q99]

print(f"Cleaned dataset rows: {len(df)}")

# Extract metadata mappings for frontend cascading dropdowns
states = sorted(df['State_Name'].unique().tolist())
seasons = sorted(df['Season'].unique().tolist())
all_crops = sorted(df['Crop'].unique().tolist())

state_districts = {}
for state in states:
    districts = sorted(df[df['State_Name'] == state]['District_Name'].unique().tolist())
    state_districts[state] = districts

district_crops = {}
for (state, district), group in df.groupby(['State_Name', 'District_Name']):
    key = f"{state}||{district}"
    district_crops[key] = sorted(group['Crop'].unique().tolist())

# Approximate baseline Mandi price per Ton (INR ₹)
mandi_prices_inr = {
    "Rice": 23000,
    "Wheat": 22750,
    "Maize": 20900,
    "Sugarcane": 3150,
    "Cotton(lint)": 65000,
    "Arhar/Tur": 70000,
    "Gram": 54000,
    "Groundnut": 63000,
    "Jowar": 3180,
    "Bajra": 2500,
    "Potato": 15000,
    "Onion": 18000,
    "Soyabean": 4600,
    "Mustard": 56500,
    "Sunflower": 67000,
    "Jute": 50500,
    "Coconut": 25000,
}

metadata = {
    "states": states,
    "seasons": seasons,
    "crops": all_crops,
    "state_districts": state_districts,
    "district_crops": district_crops,
    "mandi_prices": mandi_prices_inr
}

with open("static/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Saved metadata.json to static/metadata.json")

# Train ML Model
X = df[['State_Name', 'District_Name', 'Season', 'Crop', 'Area', 'Crop_Year']]
y = df['Yield']

# Sample 80,000 rows if dataset is large for fast training while retaining strong accuracy
if len(df) > 80000:
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=80000, random_state=42, stratify=None)
else:
    X_sample, y_sample = X, y

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['State_Name', 'District_Name', 'Season', 'Crop']),
        ('num', 'passthrough', ['Area', 'Crop_Year'])
    ]
)

# Use HistGradientBoostingRegressor or RandomForestRegressor
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(max_iter=150, random_state=42))
])

print("Training Machine Learning Model...")
model_pipeline.fit(X_sample, y_sample)

# Evaluate model
score = model_pipeline.score(X_sample, y_sample)
print(f"Model R^2 score on train sample: {score:.4f}")

joblib.dump(model_pipeline, "model.joblib", compress=3)
model_size_mb = os.path.getsize("model.joblib") / (1024 * 1024)
print(f"Successfully saved model.joblib! Size: {model_size_mb:.2f} MB")
