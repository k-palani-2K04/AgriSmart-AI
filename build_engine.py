import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import HistGradientBoostingRegressor
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

# Remove global extreme unit anomalies (Yield > 120 Tons/Ha)
df = df[df['Yield'] <= 120.0]

# Remove crop-specific extreme outliers (98th percentile per crop)
clean_dfs = []
for crop, group in df.groupby('Crop'):
    if len(group) > 5:
        q98 = group['Yield'].quantile(0.98)
        group = group[group['Yield'] <= q98]
    clean_dfs.append(group)

df = pd.concat(clean_dfs, ignore_index=True)

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

# Complete APMC Mandi benchmark prices per Ton (INR ₹) covering all crop categories
mandi_prices_inr = {
    # Cereals & Grains
    "Rice": 23000, "Paddy": 21830, "Wheat": 22750, "Maize": 20900, "Jowar": 31800,
    "Bajra": 25000, "Ragi": 38400, "Barley": 18500, "Small millets": 32000,
    "Korra": 31000, "Samai": 33000, "Varagu": 34000, "Total foodgrain": 24000,
    "Other Cereals & Millets": 22000,

    # Pulses
    "Arhar/Tur": 70000, "Gram": 54000, "Moong(Green Gram)": 85500, "Urad": 69000,
    "Blackgram": 68000, "Masoor": 64000, "Lentil": 64000, "Horse-gram": 42000,
    "Khesari": 38000, "Moth": 60000, "Cowpea(Lobia)": 52000, "Pulses total": 58000,
    "Other Kharif pulses": 55000, "Other Rabi pulses": 56000, "Other misc. pulses": 54000,
    "Peas & beans (Pulses)": 48000, "Rajmash Kholar": 75000, "Ricebean (nagadal)": 62000,

    # Oilseeds
    "Groundnut": 63000, "Sesamum": 115000, "Mustard": 56500, "Rapeseed & Mustard": 56500,
    "Soyabean": 46000, "Sunflower": 67000, "Castor seed": 61000, "Linseed": 58000,
    "Niger seed": 72000, "Safflower": 54000, "Oilseeds total": 55000, "Other oilseeds": 52000,

    # Commercial & Plantation
    "Sugarcane": 3150, "Cotton(lint)": 65000, "Kapas": 65000, "Jute": 50500,
    "Mesta": 38000, "Jute & Mesta": 45000, "Sannhamp": 32000, "Rubber": 160000,
    "Tea": 180000, "Coffee": 210000, "Tobacco": 95000, "Coconut": 25000,
    "Arecanut": 350000, "Arecanut (Raw)": 180000, "Arecanut (Processed)": 420000,
    "Cashewnut": 120000, "Cashewnut Raw": 110000, "Cashewnut Processed": 650000,

    # Spices & Condiments
    "Turmeric": 135000, "Dry ginger": 180000, "Ginger": 60000, "Garlic": 80000,
    "Dry chillies": 160000, "Coriander": 75000, "Cardamom": 1400000, "Black pepper": 550000,
    "Cond-spcs other": 70000,

    # Vegetables
    "Potato": 15000, "Onion": 18000, "Tomato": 22000, "Brinjal": 20000, "Bhindi": 24000,
    "Cabbage": 16000, "Cauliflower": 18000, "Carrot": 20000, "Radish": 14000, "Turnip": 15000,
    "Sweet potato": 18000, "Tapioca": 12000, "Yam": 22000, "Colocosia": 20000,
    "Cucumber": 18000, "Pumpkin": 14000, "Ash Gourd": 13000, "Bitter Gourd": 26000,
    "Bottle Gourd": 15000, "Snake Guard": 16000, "Ribed Guard": 18000, "Drum Stick": 30000,
    "Beans & Mutter(Vegetable)": 32000, "Peas (vegetable)": 35000, "Other Vegetables": 20000,

    # Fruits
    "Banana": 22000, "Papaya": 18000, "Mango": 45000, "Citrus Fruit": 30000, "Lemon": 35000,
    "Orange": 38000, "Other Citrus Fruit": 32000, "Grapes": 65000, "Apple": 75000,
    "Pineapple": 28000, "Pomegranate": 85000, "Sapota": 25000, "Watermelon": 12000,
    "Jack Fruit": 15000, "Peach": 50000, "Pear": 55000, "Plums": 60000, "Litchi": 70000,
    "Ber": 20000, "Other Fresh Fruits": 30000, "Other Dry Fruit": 250000
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

print("Saved updated metadata.json to static/metadata.json")

# Train ML Model
X = df[['State_Name', 'District_Name', 'Season', 'Crop', 'Area', 'Crop_Year']]
y = df['Yield']

# Sample balanced 80,000 rows across crop categories
if len(df) > 80000:
    crop_counts = df['Crop'].value_counts()
    valid_crops = crop_counts[crop_counts >= 2].index
    df_strat = df[df['Crop'].isin(valid_crops)]
    X_strat = df_strat[['State_Name', 'District_Name', 'Season', 'Crop', 'Area', 'Crop_Year']]
    y_strat = df_strat['Yield']
    X_sample, _, y_sample, _ = train_test_split(X_strat, y_strat, train_size=80000, random_state=42, stratify=df_strat['Crop'])
else:
    X_sample, y_sample = X, y

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['State_Name', 'District_Name', 'Season', 'Crop']),
        ('num', 'passthrough', ['Area', 'Crop_Year'])
    ]
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(max_iter=150, random_state=42))
])

print("Training Machine Learning Model with Stratified Crop Sampling...")
model_pipeline.fit(X_sample, y_sample)

score = model_pipeline.score(X_sample, y_sample)
print(f"Model R^2 score on train sample: {score:.4f}")

joblib.dump(model_pipeline, "model.joblib", compress=3)
model_size_mb = os.path.getsize("model.joblib") / (1024 * 1024)
print(f"Successfully saved model.joblib! Size: {model_size_mb:.2f} MB")
