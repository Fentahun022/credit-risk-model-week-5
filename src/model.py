import pandas as pd
import mlflow
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from src import config
from src.data_processing import create_proxy_target, create_features, get_preprocessor

def train_model():
    """Trains, logs, registers, and promotes the credit risk model."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI_TRAIN)
    
    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        print(f"Starting run {run.info.run_name} in experiment_id '{experiment_id}'")
        
        print("Loading and processing data...")
        df_raw = pd.read_csv(config.RAW_DATA_PATH)
        proxy_target_df = create_proxy_target(df_raw)
        features_df = create_features(df_raw)
        final_df = pd.merge(features_df, proxy_target_df, on='CustomerId')
        X = final_df.drop(columns=['CustomerId', 'is_high_risk'])
        y = final_df['is_high_risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
        
        print("Training model...")
        preprocessor = get_preprocessor()
        model = GradientBoostingClassifier(**config.MODEL_PARAMS)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)

        print("Evaluating and logging model...")
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("roc_auc", roc_auc)
        signature = infer_signature(X_train.head(1), pipeline.predict_proba(X_train.head(1)))
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=config.REGISTERED_MODEL_NAME
        )
        
        print("Searching for the latest version of the registered model...")
        client = MlflowClient()
        latest_versions = client.get_latest_versions(name=config.REGISTERED_MODEL_NAME, stages=["None"])
        if not latest_versions:
            raise Exception(f"No model versions found for '{config.REGISTERED_MODEL_NAME}' in stage 'None'.")
        model_version = latest_versions[0].version
        
        print(f"Found latest version: {model_version}. Promoting to Production...")
        client.transition_model_version_stage(
            name=config.REGISTERED_MODEL_NAME,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"Successfully promoted model version {model_version} to Production.")

# ==============================================================================
# === NEW FUNCTIONS TO ADD =====================================================
# ==============================================================================

def get_production_model():
    """Loads the production-stage model from the MLflow Model Registry."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI_API)
    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}/Production"
    return mlflow.sklearn.load_model(model_uri=model_uri)

def explain_prediction(model_pipeline, customer_data):
    """Generates a SHAP explanation for a single prediction."""
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']
    
    # Get feature names after preprocessing, handling potential scikit-learn version differences
    try:
        num_features = preprocessor.transformers_[0][2]
        cat_features_raw = preprocessor.transformers_[1][2]
        cat_features_out = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features_raw)
        feature_names = num_features + list(cat_features_out)
    except AttributeError:
        # Fallback for older scikit-learn versions or different pipeline structures
        feature_names = model_pipeline.named_steps['classifier'].feature_names_in_

    # Create the SHAP explainer
    explainer = shap.TreeExplainer(classifier)
    
    # Transform the single customer data row
    transformed_data = preprocessor.transform(customer_data)
    
    # Get SHAP values
    shap_values = explainer.shap_values(transformed_data)
    
    # Create the explanation object for plotting
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=transformed_data[0],
        feature_names=feature_names
    )
    return explanation

# ==============================================================================
# === END OF NEW FUNCTIONS =====================================================
# ==============================================================================

if __name__ == '__main__':
    # This allows you to run the training by executing `python -m src.model`
    train_model()