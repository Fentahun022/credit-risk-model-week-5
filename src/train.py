# src/train.py

import pandas as pd
import mlflow
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from data_processing import get_preprocessor

def main(processed_data_path, model_name):
    """
    Main training script. Loads processed data and trains the model.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit_Risk_B5W5_Production")

    with mlflow.start_run():
        print(f"Loading processed data from '{processed_data_path}'...")
        final_df = pd.read_csv(processed_data_path)

        print("Defining features and target...")
        X = final_df.drop(columns=['CustomerId', 'is_high_risk'])
        y = final_df['is_high_risk']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training model: {model_name}")
        preprocessor = get_preprocessor()
        model = GradientBoostingClassifier(random_state=42)
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        print("Test Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        mlflow.log_params({"model_class": model.__class__.__name__})
        mlflow.log_metrics(metrics)
        
        signature = infer_signature(X_train.head(1), pipeline.predict_proba(X_train.head(1)))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="credit_risk_model_artifact",
            signature=signature,
            registered_model_name=model_name
        )

        # --- THIS IS THE FINAL FIX ---
        # The traceback shows that registered_model_version IS the version number (as an int).
        model_version = model_info.registered_model_version
        
        print(f"Transitioning model version {model_version} to Production...")
        
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True 
        )

        print(f"Model '{model_name}' version {model_version} trained, registered, and promoted to Production.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a credit risk model using processed data.")
    parser.add_argument("--processed_data_path", type=str, default="data/processed/processed_credit_data.csv", help="Path to the processed data CSV.")
    parser.add_argument("--model_name", type=str, default="CreditRiskModel", help="Name to register the model in MLflow.")
    args = parser.parse_args()
    
    main(args.processed_data_path, args.model_name)