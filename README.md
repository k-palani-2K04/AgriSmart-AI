# AgriSmart AI Pro - Indian Crop Yield & Plant Health Platform 🌾

**AgriSmart AI Pro** is an advanced, end-to-end precision agritech platform built for Indian agriculture. It combines a Machine Learning Yield Prediction engine, APMC Mandi revenue valuation, a regional crop recommendation system, and a local **Ollama Vision AI Leaf Inspector** for plant disease diagnosis.

All land area measurements are standardized in **Acres**, crop yield in **Tons/Acre**, and financial market valuations in **Indian Rupees (INR ₹)**.

---

## 🚀 Key Features

### 1. 📊 ML Crop Yield Predictor
- **Units**: Land Area in **Acres**, Yield in **Tons/Acre**, Revenue in **INR ₹**.
- **Model**: Retrained Random Forest Regressor ($R^2 \approx 92.92\%$) trained on historical All-India crop production data.
- **Agronomic Safety Controls**: Includes category-specific yield upper-bound caps (foodgrains, sugarcane, horticultural vegetables) and outlier filtering to prevent hyper-inflated or realistic forecast anomalies.
- **Smart Two-Way Cascading Dropdowns**: Covers all 33 States and 646 Districts of India. Supports both State-first filtering and direct District-first selection with automatic State auto-detection.

### 2. 🍃 AI Leaf Inspector (Vision LLM)
- **Local Vision AI**: Powered by local Ollama Vision models (`llava` / `llama3.2-vision`) for private, offline plant health analysis.
- **Computer Vision Relevance Check**: Automatically analyzes leaf image HSV foliage pigment ($\ge 12\%$ green/yellow/brown vegetation threshold) before invoking AI inference, rejecting non-leaf uploads with clear error guidance.
- **Actionable Diagnosis**: Delivers detailed Markdown reports covering disease identification, severity assessment, organic remedies, chemical treatments, and preventative guidelines.

### 3. 💡 Optimal Regional Crop Recommender
- Ranks top-performing crops for any Indian State, District, and Season (Kharif, Rabi, Summer, Whole Year, etc.) based on predicted yield (Tons/Acre) and APMC Mandi market valuation (INR ₹).

### 4. 💰 APMC Mandi Market Rates Directory
- Contains market valuation rates (INR ₹ / Ton) covering 124 crop species.
- Features dynamic client-side live search filtering.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, Flask Framework
- **Machine Learning**: Scikit-Learn (Random Forest Regressor), Joblib, Pandas, NumPy
- **Computer Vision & AI**: OpenCV, Pillow (PIL), Ollama Vision API (`llava` / `llama3.2-vision`)
- **Frontend**: HTML5, Tailwind CSS 3, FontAwesome 6, Google Fonts (*Plus Jakarta Sans* & *Space Grotesk*), Marked.js
- **Testing**: Pytest automated test suite (100% test pass rate)

---

## 📐 Units & Standards

| Parameter | Unit | Symbol |
| :--- | :--- | :--- |
| **Land Area** | Acres | Acre |
| **Crop Yield** | Tons per Acre | Tons/Acre |
| **Total Production** | Metric Tons | Tons |
| **Market Valuation** | Indian Rupees | INR ₹ |
| **Mandi Price Rate** | Rupees per Ton | ₹ / Ton |

---

## 📁 Project Structure

```
📦 Crop-Yield-Prediction-main
├── app.py                     # Flask Web Server & API Routes
├── model_engine.py            # Agritech ML & CV Engine Logic
├── build_engine.py            # Data Cleaning, Model Training & Metadata Build Script
├── model.joblib               # Trained Random Forest Regressor Model
├── static/
│   ├── metadata.json          # Crop, District, State & Mandi Rates Metadata
│   ├── css/                   # Custom Stylesheets
│   └── gallery/               # App Branding & Static Assets
├── templates/
│   ├── index.html             # Main AgriSmart AI Pro Dashboard (Tabs 1-4)
│   └── predict.html           # Classic Prediction Form
├── tests/
│   └── test_app.py            # Pytest Automated Integration Tests
├── India_Crop_Production.csv  # Historical All-India Agriculture Dataset
├── requirements.txt           # Python Dependencies
└── README.md                  # Project Documentation
```

---

## ⚙️ Quick Start & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/BhaveshBhakta/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction
pip install -r requirements.txt
```

### 2. (Optional) Re-build ML Model & Metadata
```bash
python build_engine.py
```

### 3. Run Automated Tests
```bash
python -m pytest
```

### 4. Start the Flask Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

---

## 🔌 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | `GET` | Main AgriSmart AI Pro Dashboard |
| `POST /predict_yield` | `POST` | Calculates ML yield forecast (Tons/Acre), total production, and Mandi revenue (₹) |
| `POST /recommend_crop` | `POST` | Ranks top optimal crops for a State, District & Season |
| `POST /detect_disease` | `POST` | Analyzes uploaded leaf image and returns Vision AI health diagnosis |
| `GET /api/states` | `GET` | Returns list of available Indian States |
| `GET /api/districts/<state>` | `GET` | Returns list of districts for a given state |
| `GET /api/crops/<state>/<district>` | `GET` | Returns crops cultivated in a given state & district |
| `GET /api/mandi_prices` | `GET` | Returns APMC Mandi market valuation rates (₹/Ton) |

---

## 🤝 Contributing
Contributions, bug reports, and feature requests are welcome! Feel free to fork the repository and submit a pull request.
