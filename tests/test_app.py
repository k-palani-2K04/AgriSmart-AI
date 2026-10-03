import io
import pytest
from PIL import Image
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test that home dashboard loads with 200 OK status."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"AgriSmart AI" in response.data

def test_api_states(client):
    """Test API returning list of Indian states."""
    response = client.get('/api/states')
    assert response.status_code == 200
    data = response.get_json()
    assert "states" in data
    assert len(data["states"]) > 0

def test_api_districts(client):
    """Test API returning districts for a specific state."""
    response = client.get('/api/districts/Punjab')
    assert response.status_code == 200
    data = response.get_json()
    assert data["state"] == "Punjab"
    assert "districts" in data
    assert len(data["districts"]) > 0

def test_predict_yield_valid(client):
    """Test valid yield prediction request."""
    payload = {
        "state": "Punjab",
        "district": "LUDHIANA",
        "season": "Kharif",
        "crop": "Rice",
        "area": 5.0,
        "year": 2024
    }
    response = client.post('/predict_yield', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "yield_per_hectare" in data["data"]
    assert data["data"]["yield_per_hectare"] > 0
    assert data["data"]["total_production"] > 0
    assert data["data"]["estimated_revenue_inr"] > 0

def test_predict_yield_negative_area(client):
    """Test error handling for negative or zero land area."""
    payload = {
        "state": "Punjab",
        "district": "LUDHIANA",
        "season": "Kharif",
        "crop": "Rice",
        "area": -10.0
    }
    response = client.post('/predict_yield', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_predict_yield_missing_fields(client):
    """Test error handling when required fields are missing."""
    payload = {
        "state": "Punjab",
        "area": 5.0
    }
    response = client.post('/predict_yield', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_recommend_crop(client):
    """Test crop recommendation endpoint."""
    payload = {
        "state": "Punjab",
        "district": "LUDHIANA",
        "season": "Kharif",
        "area": 2.0
    }
    response = client.post('/recommend_crop', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0

def test_detect_disease_valid_image(client):
    """Test leaf disease endpoint with a valid generated image."""
    img = Image.new('RGB', (200, 200), color=(34, 139, 34)) # Foliage Green
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {
        'leaf_image': (img_byte_arr, 'leaf_test.jpg')
    }
    response = client.post('/detect_disease', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] in ["success", "invalid_image"]

def test_detect_disease_non_leaf_image(client):
    """Test leaf disease endpoint with a blank solid red non-foliage image to test relevance rejection."""
    img = Image.new('RGB', (200, 200), color=(255, 0, 0)) # Pure Red
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {
        'leaf_image': (img_byte_arr, 'red_box.jpg')
    }
    response = client.post('/detect_disease', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] == "invalid_image"
    assert "Invalid Image" in res_data["analysis"] or "Irrelevant Image" in res_data["analysis"]
