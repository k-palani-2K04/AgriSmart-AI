import os
from flask import Flask, request, jsonify, render_template
from model_engine import AgritechEngine

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB Max Upload Limit
engine = AgritechEngine()

@app.route("/")
def home():
    states = engine.get_states()
    seasons = engine.get_seasons()
    all_districts = engine.get_all_districts()
    district_to_state = engine.get_district_to_state_map()
    return render_template("index.html", states=states, seasons=seasons, all_districts=all_districts, district_to_state=district_to_state)

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/api/states", methods=["GET"])
def api_states():
    return jsonify({"states": engine.get_states()})

@app.route("/api/districts/<state>", methods=["GET"])
def api_districts(state):
    districts = engine.get_districts(state)
    return jsonify({"state": state, "districts": districts})

@app.route("/api/crops/<state>/<district>", methods=["GET"])
def api_crops(state, district):
    crops = engine.get_crops(state, district)
    return jsonify({"state": state, "district": district, "crops": crops})

@app.route("/api/mandi_prices", methods=["GET"])
def api_mandi_prices():
    prices = engine.metadata.get("mandi_prices", {})
    return jsonify({"status": "success", "mandi_prices": prices})

@app.route("/predict", methods=["GET", "POST"])
@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        state = data.get("state") or data.get("State_Name")
        district = data.get("district") or data.get("District_Name")
        season = data.get("season") or data.get("Season")
        crop = data.get("crop") or data.get("Crop")
        area_str = data.get("area") or data.get("Area")
        year_str = data.get("year") or data.get("Crop_Year", 2024)

        if not all([state, district, season, crop, area_str]):
            return jsonify({"error": "Missing required fields: state, district, season, crop, and area are required."}), 400

        try:
            area = float(area_str)
            if area <= 0:
                return jsonify({"error": "Area must be a positive number of Acres greater than 0."}), 400
        except ValueError:
            return jsonify({"error": "Invalid Area value. Must be a numeric number."}), 400

        try:
            year = int(year_str)
        except ValueError:
            year = 2024

        result = engine.predict_yield(state, district, season, crop, area, year)

        if request.path == "/predict" and not request.is_json:
            return render_template("predict.html", prediction=f"{result['yield_per_acre']} Tons/Acre (Total: {result['total_production']} Tons, Mandi Rev: {result['formatted_revenue']})")

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/recommend_crop", methods=["POST"])
def recommend_crop():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        state = data.get("state")
        district = data.get("district")
        season = data.get("season")
        area_str = data.get("area", 1.0)

        if not all([state, district, season]):
            return jsonify({"error": "Missing required fields: state, district, and season."}), 400

        try:
            area = float(area_str)
            if area <= 0:
                area = 1.0
        except ValueError:
            area = 1.0

        recommendations = engine.recommend_crops(state, district, season, area)
        return jsonify({"status": "success", "recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/detect_disease", methods=["POST"])
def detect_disease():
    try:
        if 'leaf_image' not in request.files:
            return jsonify({"error": "No leaf_image file uploaded."}), 400

        file = request.files['leaf_image']
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400

        image_bytes = file.read()
        result = engine.detect_leaf_disease(image_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Disease Detection Failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
