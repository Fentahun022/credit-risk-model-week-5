from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"

def test_valid_prediction():
    payload = {"total_transactions": 10, "total_value": 50000, "avg_value": 5000, "std_value": 1500, "unique_products": 3, "most_frequent_channel": "Channel_Id_4"}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        assert response.json() == {"detail": "Model is not available. Please check server logs."}
    else:
        assert response.status_code == 200
        data = response.json()
        assert "risk_probability" in data
        assert "is_high_risk" in data
        assert 0 <= data["risk_probability"] <= 1

def test_invalid_prediction_input():
    payload = {"total_transactions": "ten", "total_value": 50000, "avg_value": 5000, "std_value": 1500, "unique_products": 3, "most_frequent_channel": "Channel_Id_4"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422