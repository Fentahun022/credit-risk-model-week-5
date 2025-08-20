import os
import mlflow
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# Import from our refactored project structure
from src import config
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

# --- Global variable for the model ---
# This will be populated during the application's lifespan startup
model = None

def load_model():
    """
    Loads the model from the MLflow Model Registry.
    This function is called by the lifespan event handler.
    """
    global model
    
    # Set the tracking URI to connect to the correct database
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI_API))
    
    # This env var is set in docker-compose to help resolve artifact paths inside the container
    if os.getenv("MLFLOW_ARTIFACT_ROOT"):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_ARTIFACT_ROOT")
    
    model_name = os.getenv("MODEL_NAME", config.REGISTERED_MODEL_NAME)
    model_stage = "Production"
    
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        print(f"Successfully loaded model '{model_name}' stage '{model_stage}'.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# --- Lifespan Event Handler (Modern FastAPI) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    The model is loaded on startup.
    """
    print("--- API startup ---")
    load_model()
    yield
    print("--- API shutdown ---")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict credit risk probability for a customer.",
    version="2.0.0",
    lifespan=lifespan  # Use the modern lifespan event handler
)

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint for health check."""
    status = "Model loaded successfully." if model is not None else "Error: Model not loaded. Check server startup logs."
    return {"status": "API is running", "model_status": status}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    """Accepts customer features and returns a risk probability."""
    if model is None:
        # This error message now matches the one in the integration test
        raise HTTPException(status_code=503, detail="Model is not available. Please check server startup logs.")

    input_df = pd.DataFrame([features.dict()])
    
    try:
        risk_probability = model.predict_proba(input_df)[0, 1]
        is_high_risk = model.predict(input_df)[0]
        return PredictionResponse(risk_probability=risk_probability, is_high_risk=int(is_high_risk))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")