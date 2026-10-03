# Indian Crop Yield Prediction Using Machine Learning

This repository contains a machine learning-based solution to predict crop yield in India, considering multiple input parameters. The project leverages a Random Forest Regressor model and includes an interactive and visually appealing website to enhance user experience.


## Project Overview

Indian agriculture depends heavily on precise crop yield predictions. This project helps farmers, policymakers, and agricultural stakeholders make informed decisions by predicting crop yield based on key input parameters. 

### Features:
- **Extensive Data**: Covers all states and over 600 districts of India.
- **User Inputs**: Crop Type, Season, Area, Crop Year, and more.
- **Machine Learning**: Random Forest Regressor for accurate yield predictions.
- **Dynamic Website**: A user-friendly interface featuring animations using GSAP and smooth scrolling powered by Lenis.


## Tech Stack

### Backend:
- **Machine Learning**: Random Forest Regressor
- **Python Libraries**: Pandas, NumPy, Scikit-learn

### Frontend:
- **UI Enhancements**: GSAP for animations and Lenis for smooth scrolling
- **Web Framework**: Flask
- **HTML/CSS/JavaScript**: For responsive and dynamic user interface


## How It Works

1. **Input Parameters**: Users input crop type, season, area, crop year, and other relevant details.
2. **Prediction Model**: The Random Forest Regressor processes the input and predicts the crop yield.
3. **Output**: The predicted yield is displayed on the website in a clean and interactive format.


## Highlights

- **Comprehensive Data**: A wide dataset ensures accurate predictions for different regions and conditions.
- **Intuitive Design**: The website combines functionality with aesthetics for a seamless user experience.
- **Dynamic Animations**: GSAP and Lenis make the interface engaging and modern.


## Setup Instructions

1. Clone this repository:  
   ```bash
   git clone https://github.com/BhaveshBhakta/Crop-Yield-Prediction.git
   cd Crop-Yield-Prediction
   ```

2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:  
   ```bash
   python app.py
   ```

4. Open the app in your browser at `http://127.0.0.1:5000/`.


## Project Structure

```
📦 indian-crop-yield-prediction
├── app.py                # Flask application
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```


## Dataset

The dataset includes:
- **State-level Data**: Comprehensive data for all Indian states.
- **District-level Data**: Covers over 600 districts for granular analysis.

## Future Roadmap & Upcoming Features 🚀

### 1. 🍃 Local Ollama Vision LLM for Leaf Disease Detection
- **Offline & Private AI**: Integrate local Ollama Vision models (e.g., `llama3.2-vision` or `llava`) to analyze uploaded crop leaf images.
- **Automated Diagnosis & Remedies**: Provide instant plant disease diagnosis, severity assessment, organic & chemical remedies, and preventative guidelines directly from local AI inference without requiring external cloud API keys or internet connection.

### 2. 🌤️ Real-Time Agrometeorology (Weather API Integration)
- Fetch live precipitation, temperature, humidity, and solar radiation via Open-Meteo API.

### 3. 💡 Smart Crop Optimization & Recommendation Engine
- Recommend high-yield, drought-resistant crops based on district soil properties and climate forecasts.

### 4. 💰 Mandi APMC Live Market Price & Profitability Calculator
- Connect with Agmarknet API to calculate estimated revenue and net profit per hectare.

### 5. 🗣️ Multilingual Voice & AI Assistant
- Provide voice-based guidance in Indian regional languages (Hindi, Marathi, Punjabi, Tamil, Telugu) for local farmers.

## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## Website Overview

![Cypa](https://github.com/user-attachments/assets/6fe9421c-c27b-4028-9897-20e1fbf09b29)

![Screenshot 2024-11-26 222147](https://github.com/user-attachments/assets/c50696b6-10e3-41cc-a83d-83bd9212c835)

![Screenshot 2024-11-26 222210](https://github.com/user-attachments/assets/6e93cb8a-ab02-4102-8912-1e4067825d15)

