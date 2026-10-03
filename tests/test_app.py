import io
import pytest
import numpy as np
from PIL import Image
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"AgriSmart AI" in response.data

def test_api_states(client):
    response = client.get('/api/states')
    assert response.status_code == 200
    data = response.get_json()
    assert "states" in data
    assert len(data["states"]) > 0

def test_api_districts(client):
    response = client.get('/api/districts/Punjab')
    assert response.status_code == 200
    data = response.get_json()
    assert data["state"] == "Punjab"
    assert "districts" in data
    assert len(data["districts"]) > 0

def test_api_mandi_prices(client):
    response = client.get('/api/mandi_prices')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "mandi_prices" in data
    assert "Rice" in data["mandi_prices"]

def test_predict_yield_valid(client):
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
    assert "yield_per_acre" in data["data"]
    assert data["data"]["yield_unit"] == "Tons / Acre"
    assert data["data"]["yield_per_acre"] > 0

def test_predict_yield_negative_area(client):
    payload = {
        "state": "Punjab",
        "district": "LUDHIANA",
        "season": "Kharif",
        "crop": "Rice",
        "area": -10.0
    }
    response = client.post('/predict_yield', json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_recommend_crop(client):
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
    assert len(data["recommendations"]) > 0

def test_detect_disease_healthy_leaf(client):
    img = Image.new('RGB', (200, 200), color=(34, 139, 34))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {'leaf_image': (img_byte_arr, 'healthy_leaf.jpg')}
    response = client.post('/detect_disease', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] == "success"
    assert "HEALTHY LEAF" in res_data["analysis"]

def test_detect_disease_diseased_leaf(client):
    img = Image.new('RGB', (200, 200), color=(34, 139, 34))
    img_np = np.array(img)
    img_np[40:140, 40:140] = [100, 50, 20] # Necrotic brown spot area
    img_np[140:190, 140:190] = [220, 190, 30] # Yellow halo area
    img_spotted = Image.fromarray(img_np)
    
    img_byte_arr = io.BytesIO()
    img_spotted.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {'leaf_image': (img_byte_arr, 'spotted_leaf.jpg')}
    response = client.post('/detect_disease', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] == "success"
    assert ("DISEASED LEAF" in res_data["analysis"] or "CHLOROSIS" in res_data["analysis"])

def test_detect_disease_irrelevant_image(client):
    img = Image.new('RGB', (200, 200), color=(255, 0, 0)) # Pure Red
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {'leaf_image': (img_byte_arr, 'red_box.jpg')}
    response = client.post('/detect_disease', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] == "invalid_image"
    assert any(term in res_data["analysis"].lower() for term in ["irrelevant", "invalid", "not contain", "foliage", "lacks"])
