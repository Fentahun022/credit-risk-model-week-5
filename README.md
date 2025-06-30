# B5W5: Credit Risk Model for Alternative Data

This project implements an end-to-end credit risk model for Bati Bank's new Buy-Now-Pay-Later service, using transactional data from an eCommerce partner. It covers the full lifecycle from data exploration and feature engineering to model training, deployment via a containerized API, and CI/CD automation.

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord mandates that financial institutions maintain capital reserves proportional to the risk they undertake. A key component of this is the Internal Ratings-Based (IRB) approach, which allows banks to use their own internal models to calculate credit risk. For regulators to approve such a model, it must be transparent, robust, and well-documented. An interpretable model, like Logistic Regression with Weight of Evidence (WoE), allows the bank to clearly explain to regulators *why* the model assigns a certain risk score to a customer based on specific features. A "black box" model, even if more accurate, poses a significant regulatory challenge because its decision-making process is opaque. Therefore, Basel II's requirements drive us towards models that are not only predictive but also explainable and auditable.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

We are using alternative data (e-commerce transactions) which does not contain a historical "loan default" label. To train a supervised machine learning model, we need a target variable to learn from. Creating a **proxy variable** is a necessary workaround. We hypothesize that customer behavior, specifically disengagement (high recency, low frequency, and low monetary value), is a strong indicator of future financial unreliability and thus a proxy for default risk.

# Potential Business Risks:
1.  Correlation vs. Causation: Our proxy for "high risk" is customer disengagement. While plausible, this might not perfectly correlate with actual loan default. We risk building a model that is good at predicting customer churn but not default.
2.  False Positives: We might incorrectly label a valuable, engaged customer as "high risk" if their purchasing pattern changes, leading to denying them credit and losing their business.
3.  False Negatives: We might label a genuinely high-risk individual as "low risk" because they are an active shopper, leading to loan defaults and financial loss. The proxy must be continuously validated against actual repayment behavior once the service is launched.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The primary trade-off is between **Performance and Interpretability/Compliance**.

 Feature  Simple Model (e.g., Logistic Regression)  Complex Model (e.g., Gradient Boosting)

**Performance**  Generally lower predictive power.  Typically higher accuracy, ROC-AUC, etc. 
 **Interpretability** 
  **High.** Coefficients are directly explainable. Easy to explain to regulators and business stakeholders why a decision was made.  
 **Low.** A "black box." Difficult to explain the contribution of individual features to a specific prediction. Requires techniques like SHAP for post-hoc explanation. 
**Regulatory Risk** 
 **Low.** Meets the transparency and auditability requirements of frameworks like Basel II. 
  **High.** Regulators may be skeptical of models they cannot easily understand, potentially delaying or denying approval. 

**Implementation** Simpler to implement, debug, and maintain. 
 More complex to tune and prone to overfitting if not handled carefully. 

**Conclusion:** For a regulated financial product, it's often prudent to start with a simpler, interpretable model. Its performance can serve as a benchmark. A complex model might be used in parallel or as a "challenger," but the simpler model often remains the primary one for regulatory reporting.