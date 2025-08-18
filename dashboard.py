import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.model import get_production_model, explain_prediction
from src import config

# --- Page Configuration ---
st.set_page_config(
    page_title="Bati Bank Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
# The line st.set_option('deprecation.showPyplotGlobalUse', False) has been removed as it is obsolete in modern Streamlit versions.

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading financial risk model...")
def load_model():
    """Load the production model from the MLflow Registry. Caching ensures it's loaded only once."""
    try:
        return get_production_model()
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure a model is trained and in 'Production'. You can train a model by running `python -m src.model`.")
        return None

model = load_model()

# --- Helper Function ---
def probability_to_score(prob, min_score, max_score):
    """Maps a probability (0-1) to a standard credit score range (e.g., 300-850)."""
    # Lower probability of being high-risk should result in a higher score.
    score = min_score + (1 - prob) * (max_score - min_score)
    return int(score)

# --- UI Layout ---
st.title("🏦 Bati Bank - BNPL Credit Risk Assessment")
st.markdown("This interactive dashboard provides real-time credit risk assessments for Bati Bank's **Buy Now, Pay Later** service. It leverages a machine learning model trained on customer transactional behavior.")

# --- Input Form in Sidebar ---
with st.sidebar:
    st.header("Customer Details")
    st.markdown("Enter the customer's transactional data to generate a risk profile.")
    with st.form("customer_input_form"):
        total_transactions = st.slider("Total Transactions", 1, 100, 10, help="Total number of transactions the customer has made.")
        total_value = st.number_input("Total Transaction Value", min_value=0, value=50000, help="Sum of the value of all transactions.")
        avg_value = st.number_input("Average Transaction Value", min_value=0, value=5000, help="The average value per transaction.")
        std_value = st.number_input("Std. Dev. of Transaction Value", min_value=0, value=1500, help="The standard deviation of their transaction values.")
        unique_products = st.slider("Number of Unique Products", 1, 50, 3, help="How many different products have they purchased?")
        most_frequent_channel = st.selectbox("Most Frequent Channel", ["Channel_Id_1", "Channel_Id_2", "Channel_Id_3", "Channel_Id_4", "Channel_Id_5", "Channel_Id_6"], help="The customer's most used transaction channel.")
        
        submit_button = st.form_submit_button(label="Assess Risk")

# --- Main Panel: Display Results ---
if model is not None and submit_button:
    # Prepare data for model prediction in a DataFrame
    customer_data = pd.DataFrame([{"total_transactions": total_transactions, "total_value": total_value, "avg_value": avg_value, "std_value": std_value, "unique_products": unique_products, "most_frequent_channel": most_frequent_channel}])
    
    st.header("Risk Assessment Results")
    col1, col2 = st.columns([1, 2]) # Make the explanation column wider

    with col1:
        st.subheader("Credit Score & Recommendation")
        # Get prediction from the loaded model
        risk_proba = model.predict_proba(customer_data)[0, 1]
        credit_score = probability_to_score(risk_proba, config.MIN_SCORE, config.MAX_SCORE)

        # Display key metrics
        st.metric("Credit Score", f"{credit_score}")
        st.metric("High-Risk Probability", f"{risk_proba:.2%}")

        # Provide a clear, color-coded recommendation based on the score
        if credit_score < 580:
            st.error("#### Recommendation: **Decline**\nHigh risk detected based on behavioral patterns.")
        elif 580 <= credit_score < 670:
            st.warning("#### Recommendation: **Manual Review**\nModerate risk detected. Further diligence is advised.")
        else:
            st.success("#### Recommendation: **Approve**\nLow risk detected. Customer profile is strong.")
    
    with col2:
        st.subheader("Prediction Explanation (via SHAP)")
        with st.spinner("Generating explanation plot..."):
            explanation = explain_prediction(model, customer_data)
            
            # Create and display the SHAP waterfall plot
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.title("Feature Contribution to Risk Score", fontsize=14)
            st.pyplot(fig, bbox_inches='tight', use_container_width=True)
            st.caption("This waterfall plot shows the features that pushed the risk prediction **higher (in red)** or **lower (in blue)**, starting from the baseline. The final prediction is the value at the top.")

elif model is None:
    st.error("Model is not loaded. Cannot perform assessment. Please check the application logs or train a model.")
else:
    st.info("Please enter customer details in the sidebar and click 'Assess Risk' to generate their credit profile.")