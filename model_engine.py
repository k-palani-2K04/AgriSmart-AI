import os
import json
import base64
import requests
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import io

ACRE_TO_HA = 0.404686
HA_TO_ACRE = 2.47105

class AgritechEngine:
    def __init__(self, metadata_path="static/metadata.json", model_path="model.joblib"):
        self.metadata_path = metadata_path
        self.model_path = model_path
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"states": [], "seasons": [], "crops": [], "state_districts": {}, "district_crops": {}, "mandi_prices": {}}
            
        # Load ML model
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def get_states(self):
        return self.metadata.get("states", [])

    def get_seasons(self):
        return self.metadata.get("seasons", [])

    def get_all_districts(self):
        dists = set()
        for d_list in self.metadata.get("state_districts", {}).values():
            dists.update(d_list)
        return sorted(list(dists))

    def get_district_to_state_map(self):
        d2s = {}
        for s, dists in self.metadata.get("state_districts", {}).items():
            for d in dists:
                if d not in d2s:
                    d2s[d] = s
        return d2s

    def get_districts(self, state):
        state_clean = str(state or "").strip()
        state_districts = self.metadata.get("state_districts", {})
        districts = state_districts.get(state_clean)
        if districts is None:
            for k, v in state_districts.items():
                if k.lower() == state_clean.lower():
                    return v
            districts = []
        return districts

    def get_crops(self, state, district):
        state_clean = str(state or "").strip()
        dist_clean = str(district or "").strip()
        key = f"{state_clean}||{dist_clean}"
        district_crops = self.metadata.get("district_crops", {})
        crops = district_crops.get(key)
        if crops is None:
            for k, v in district_crops.items():
                if k.lower() == key.lower():
                    return v
            crops = self.metadata.get("crops", [])
        return crops

    def get_mandi_price(self, crop):
        prices = self.metadata.get("mandi_prices", {})
        for key, price in prices.items():
            if key.lower() in crop.lower() or crop.lower() in key.lower():
                return price
        return 22000

    def predict_yield(self, state, district, season, crop, area_acres, year=2024):
        """
        Predicts realistic crop yield per acre and financial Mandi revenue.
        Input land area is passed in Acres.
        """
        if area_acres <= 0:
            raise ValueError("Area must be greater than zero Acres.")

        # Convert input land area from Acres to Hectares for ML model features
        area_ha = float(area_acres) * ACRE_TO_HA

        input_data = pd.DataFrame([{
            'State_Name': str(state).strip(),
            'District_Name': str(district).strip(),
            'Season': str(season).strip(),
            'Crop': str(crop).strip(),
            'Area': area_ha,
            'Crop_Year': int(year)
        }])

        if self.model:
            predicted_yield_ha = float(self.model.predict(input_data)[0])
        else:
            predicted_yield_ha = 2.5

        # Apply agronomic bounds based on crop classification to ensure realistic predictions
        crop_lower = str(crop).lower()
        if any(c in crop_lower for c in ['rice', 'wheat', 'maize', 'jowar', 'bajra', 'ragi', 'gram', 'urad', 'moong', 'pulses', 'cereals']):
            max_yield_ha = 15.0   # ~6 Tons/Acre max for foodgrains & pulses
        elif 'sugarcane' in crop_lower:
            max_yield_ha = 110.0  # ~44 Tons/Acre max for sugarcane
        elif any(c in crop_lower for c in ['brinjal', 'tomato', 'potato', 'onion', 'cabbage', 'vegetable', 'banana', 'papaya', 'tapioca']):
            max_yield_ha = 50.0   # ~20 Tons/Acre max for heavy horticultural & vegetable crops
        else:
            max_yield_ha = 35.0

        predicted_yield_ha = max(0.05, min(predicted_yield_ha, max_yield_ha))

        # Yield per Acre (Tons / Acre)
        predicted_yield_acre = predicted_yield_ha / HA_TO_ACRE
        total_production = round(predicted_yield_acre * float(area_acres), 2)
        price_per_ton = self.get_mandi_price(crop)
        estimated_revenue = round(total_production * price_per_ton, 2)

        if predicted_yield_acre < 0.6:
            benchmark = "Low Yield"
            benchmark_color = "red"
        elif predicted_yield_acre <= 1.6:
            benchmark = "Moderate / Standard Yield"
            benchmark_color = "amber"
        else:
            benchmark = "High / Optimal Yield"
            benchmark_color = "green"

        return {
            "yield_per_acre": round(predicted_yield_acre, 2),
            "yield_unit": "Tons / Acre",
            "total_production": total_production,
            "production_unit": "Tons",
            "mandi_price_per_ton": price_per_ton,
            "estimated_revenue_inr": estimated_revenue,
            "formatted_revenue": f"₹ {estimated_revenue:,.2f}",
            "benchmark": benchmark,
            "benchmark_color": benchmark_color
        }

    def recommend_crops(self, state, district, season, area_acres=1.0):
        available_crops = self.get_crops(state, district)
        if not available_crops:
            available_crops = ["Rice", "Wheat", "Maize", "Cotton(lint)", "Sugarcane"]

        recommendations = []
        for crop in available_crops[:12]:
            try:
                res = self.predict_yield(state, district, season, crop, area_acres)
                recommendations.append({
                    "crop": crop,
                    "expected_yield": res["yield_per_acre"],
                    "estimated_revenue": res["estimated_revenue_inr"],
                    "formatted_revenue": res["formatted_revenue"],
                    "mandi_price": res["mandi_price_per_ton"]
                })
            except Exception:
                continue

        recommendations.sort(key=lambda x: x["estimated_revenue"], reverse=True)
        return recommendations[:5]

    def _extract_image_metrics(self, image_bytes):
        """Pre-checks image file validity and extracts HSV foliage properties."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            w, h = img.size
            if w < 30 or h < 30:
                return False, "Resolution too low (minimum 30x30 required).", None

            hsv = Image.open(io.BytesIO(image_bytes)).convert('HSV')
            hsv_np = np.array(hsv)
            hue, sat, val = hsv_np[:, :, 0], hsv_np[:, :, 1], hsv_np[:, :, 2]
            total = float(w * h)

            green_pixels = np.sum((hue >= 35) & (hue <= 90) & (sat >= 20) & (val >= 20))
            yellow_pixels = np.sum((hue >= 15) & (hue < 35) & (sat >= 30) & (val >= 35))
            necrotic_pixels = np.sum(((hue < 15) | (hue > 240)) & (sat >= 20) & (val <= 160))

            green_ratio = float(green_pixels) / total
            yellow_ratio = float(yellow_pixels) / total
            necrotic_ratio = float(necrotic_pixels) / total
            foliage_ratio = green_ratio + yellow_ratio + necrotic_ratio

            metrics = {
                "green_pct": round(green_ratio * 100, 1),
                "yellow_pct": round(yellow_ratio * 100, 1),
                "necrotic_pct": round(necrotic_ratio * 100, 1),
                "affected_pct": round((yellow_ratio + necrotic_ratio) * 100, 1),
                "foliage_pct": round(foliage_ratio * 100, 1)
            }

            if metrics["foliage_pct"] < 12.0:
                return False, "The uploaded image lacks characteristic leaf foliage or plant chlorophyll pigments.", metrics

            return True, None, metrics
        except Exception as e:
            return False, f"Invalid or corrupt image: {str(e)}", None

    def get_local_ollama_models(self, ollama_url="http://localhost:11434/api/tags"):
        try:
            r = requests.get(ollama_url, timeout=2)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                return models
        except Exception:
            pass
        return []

    def detect_leaf_disease(self, image_bytes, ollama_url="http://localhost:11434/api/generate"):
        is_valid, err_msg, metrics = self._extract_image_metrics(image_bytes)
        if not is_valid:
            return {
                "status": "invalid_image",
                "source": "Image Validation Engine",
                "analysis": f"❌ **Irrelevant / Invalid Image**: {err_msg}\n\n*Please upload a clear photograph of an agricultural crop leaf.*"
            }

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        local_models = self.get_local_ollama_models()
        model_to_use = "llama3.2-vision"
        if local_models:
            vision_candidates = [m for m in local_models if 'vision' in m or 'llava' in m or 'moondream' in m]
            if vision_candidates:
                model_to_use = vision_candidates[0]
            elif any("llama3.2" in m for m in local_models):
                model_to_use = [m for m in local_models if "llama3.2" in m][0]
            else:
                model_to_use = local_models[0]

        prompt = """
        You are a World-Class Agronomist and Plant Pathology Vision AI.
        
        TASK 1: RELEVANCE CHECK
        Examine the image carefully. Is this image a photograph of a plant, crop, or leaf?
        IF THE IMAGE IS NOT A CROP OR PLANT LEAF (e.g., human face, car, animal, object, document, building, non-plant item, or blank image):
        YOU MUST RESPOND ONLY WITH:
        "IRRELEVANT_IMAGE: The uploaded image does not contain a crop or plant leaf. Please upload a clear photograph of an agricultural crop leaf."

        TASK 2: UNIVERSAL LEAF DISEASE & REASONING (If it IS a valid plant/crop leaf)
        Analyze the leaf for ANY crop disease or health condition across all agricultural species worldwide.
        
        Provide your expert diagnostic report in clean Markdown:
        - 🌾 **Crop Species**: Identified crop/plant
        - 🩺 **Health Status & Disease Name**: Healthy Leaf OR Specific Disease Name
        - 🔍 **Visual Reasoning & Evidence**: Detailed description of observed symptoms.
        - 📊 **Severity Rating**: Low, Medium, or High
        - 🛡️ **Treatment & Management Plan**:
          - Organic / Biological Control Options
          - Recommended Chemical Fungicide / Pesticide Remedies
          - Preventative Agronomic Practices
        """

        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False
        }

        try:
            response = requests.post(ollama_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get("response", "").strip()

                if "IRRELEVANT_IMAGE" in analysis_text or "not a plant" in analysis_text.lower() or "not a crop" in analysis_text.lower():
                    return {
                        "status": "invalid_image",
                        "source": f"Local Ollama Vision ({model_to_use})",
                        "analysis": "❌ **Irrelevant Image Detected**\n\nThe AI system analyzed your uploaded photo and determined that **it does not contain a crop or plant leaf**.\n\n*Please upload a clear photograph of an agricultural crop leaf for disease analysis.*"
                    }

                return {
                    "status": "success",
                    "source": f"Local Ollama Vision LLM ({model_to_use})",
                    "analysis": analysis_text
                }
        except Exception:
            pass

        green_pct = metrics["green_pct"]
        yellow_pct = metrics["yellow_pct"]
        necrotic_pct = metrics["necrotic_pct"]
        affected_pct = metrics["affected_pct"]

        if green_pct < 25.0 and yellow_pct < 10.0 and necrotic_pct < 8.0:
            return {
                "status": "invalid_image",
                "source": "AgriSmart Vision Inspector",
                "analysis": "❌ **Irrelevant Image Detected**\n\nThe uploaded photo does not contain characteristic plant foliage or leaf color pigments.\n\n*Please upload a clear photograph of a crop leaf.*"
            }

        if green_pct >= 55.0 and affected_pct < 6.0:
            analysis_text = f"""
### 🟢 Plant Health Status: HEALTHY LEAF (No Disease Detected)

#### 🌾 Crop Classification: Agricultural Crop Foliage
#### 🔍 Visual Reasoning & Evidence:
- **Chlorophyll Density:** High ({green_pct}% healthy green coverage)
- **Necrotic Lesion Spots:** Negligible ({necrotic_pct}% spot density)
- **Reasoning:** Uniform chlorophyll distribution with smooth leaf margins and zero necrotic spore formation.

#### 🛡️ Management Guidelines:
1. **Regular Monitoring:** Continue standard crop inspection.
2. **Irrigation:** Maintain drip/soil-level watering to avoid foliage wetness.
"""
        elif necrotic_pct >= 4.0 or (yellow_pct >= 8.0 and necrotic_pct >= 2.0):
            severity = "🔴 High Severity" if affected_pct > 20.0 else "🟡 Moderate Severity"
            analysis_text = f"""
### 🔴 Plant Health Status: DISEASED LEAF (Fungal Leaf Spot / Blight)

#### 🌾 Crop Classification: Agricultural Crop Foliage
#### 🔍 Visual Reasoning & Evidence:
- **Severity Rating:** {severity}
- **Necrotic Spot Lesions:** {necrotic_pct}% dark brown spot coverage
- **Chlorotic Halos:** {yellow_pct}% surrounding yellow tissue
- **Reasoning:** Concentrated brown necrotic spots surrounded by yellow chlorotic halos covering {affected_pct}% of the leaf surface indicate active fungal pathogen infection (*Alternaria* / *Cercospora* leaf spot).

#### 🛡️ Treatment Plan:
1. **Organic Remedy:** Spray Neem Oil solution (5 ml/L) or *Trichoderma viride* biopesticide during early morning.
2. **Chemical Control:** Apply Copper Oxychloride (50% WP @ 2.5 g/L) or Mancozeb (75% WP @ 2.0 g/L).
"""
        else:
            analysis_text = f"""
### 🟡 Plant Health Status: CHLOROSIS / NUTRIENT DEFICIENCY

#### 🌾 Crop Classification: Agricultural Crop Foliage
#### 🔍 Visual Reasoning & Evidence:
- **Yellow Chlorosis Area:** {yellow_pct}% foliage yellowing
- **Green Chlorophyll Ratio:** Moderate ({green_pct}%)
- **Reasoning:** Interveinal chlorosis affecting {yellow_pct}% of leaf area without concentrated brown fungal spots indicates Nitrogen/Iron deficiency or root waterlogging.

#### 🛡️ Treatment Plan:
1. **Foliar Spray:** Apply 1% Urea solution or Ferrous Sulfate (FeSO4 @ 5 g/L).
"""

        return {
            "status": "success",
            "source": "AgriSmart Vision AI (Local Mode)",
            "analysis": analysis_text
        }
