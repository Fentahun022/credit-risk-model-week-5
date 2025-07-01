from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""
    total_transactions: int = Field(..., example=10, description="Total number of transactions made by the customer.")
    total_value: float = Field(..., example=50000.0, description="Total monetary value of all transactions.")
    avg_value: float = Field(..., example=5000.0, description="Average monetary value per transaction.")
    std_value: float = Field(..., example=1500.0, description="Standard deviation of transaction values.")
    unique_products: int = Field(..., example=3, description="Number of unique products purchased.")
    most_frequent_channel: str = Field(..., example="Channel_Id_4", description="The most used transaction channel.")

class PredictionResponse(BaseModel):
    """Response model for a prediction."""
    risk_probability: float = Field(..., example=0.85, description="The predicted probability of the customer being high-risk (0 to 1).")
    is_high_risk: int = Field(..., example=1, description="Binary prediction: 1 if high-risk, 0 if low-risk.")