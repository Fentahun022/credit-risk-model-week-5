# src/process_data.py

import pandas as pd
import argparse
import os

# Import our custom feature engineering functions
from data_processing import create_proxy_target, create_features

def main(raw_data_path, processed_data_dir):
    """
    This script processes the raw data and saves the final model-ready dataset.
    """
    print("Starting data processing...")
    
    # Ensure the output directory exists
    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"Output directory '{processed_data_dir}' is ready.")

    # Load raw data
    print(f"Loading raw data from '{raw_data_path}'...")
    df_raw = pd.read_csv(raw_data_path)

    # 1. Create the proxy target variable using RFM clustering
    print("Step 1: Engineering proxy target variable (is_high_risk)...")
    proxy_target_df = create_proxy_target(df_raw)

    # 2. Create customer-level aggregate features
    print("Step 2: Engineering customer-level features...")
    features_df = create_features(df_raw)
    
    # 3. Merge features and the target variable
    print("Step 3: Merging features and target variable...")
    final_df = pd.merge(features_df, proxy_target_df, on='CustomerId', how='inner')
    print(f"Final dataset shape: {final_df.shape}")
    print("Final dataset columns:", final_df.columns.tolist())
    
    # 4. Save the processed data
    output_path = os.path.join(processed_data_dir, "processed_credit_data.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"Data processing complete. Processed data saved to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data to create a training-ready dataset.")
    parser.add_argument("--raw_data_path", type=str, default="data/raw/data.csv", help="Path to the raw input data CSV.")
    parser.add_argument("--processed_data_dir", type=str, default="data/processed", help="Directory to save the processed data CSV.")
    args = parser.parse_args()
    
    main(args.raw_data_path, args.processed_data_dir)