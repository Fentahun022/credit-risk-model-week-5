import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Import our Pydantic models
from .pydantic_models import CustomerFeatures, PredictionResponse

# Load environment variables from .env file
load_dotenv()

# Set up MLflow tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict credit risk probability for a customer based on transactional behavior.",
    version="1.0.0"
)

# Load the model from MLflow Model Registry at startup
MODEL_NAME = os.getenv("MODEL_NAME", "CreditRiskModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        print(f"Successfully loaded model '{MODEL_NAME}' version '{MODEL_STAGE}' from '{model_uri}'.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # The app will run but /predict will fail. This is intentional to allow health checks.
        model = None

@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint for health check."""
    status = "Model loaded successfully." if model is not None else "Error: Model not loaded."
    return {"status": "API is running", "model_status": status}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    """
    Accepts customer features and returns a risk probability.
    
    The input should be a JSON object with the following keys:
    - `total_transactions`: int
    - `total_value`: float
    - `avg_value`: float
    - `std_value`: float
    - `unique_products`: int
    - `most_frequent_channel`: str
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check the logs.")

    # Convert Pydantic model to a pandas DataFrame for prediction
    input_df = pd.DataFrame([features.dict()])
    
    try:
        # Predict probability (we want the probability of the positive class '1')
        risk_probability = model.predict_proba(input_df)[0, 1]
        is_high_risk = model.predict(input_df)[0]
        
        return PredictionResponse(
            risk_probability=risk_probability,
            is_high_risk=int(is_high_risk)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")