import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def create_proxy_target(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Calculates RFM metrics and uses K-Means clustering to create a high-risk proxy target.
    A customer is high-risk if they belong to the cluster with the highest recency (least recent).
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + dt.timedelta(days=1)

    rfm_df = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Value': 'Monetary'
    })

    # Log transform to handle skewness
    rfm_log = rfm_df.apply(lambda x: np.log(x + 1))
    
    # Scale data for clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=random_state, n_init=10)
    kmeans.fit(rfm_scaled)
    rfm_df['Cluster'] = kmeans.labels_

    # Identify high-risk cluster (highest recency, lowest frequency/monetary)
    cluster_analysis = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster_id = cluster_analysis['Recency'].idxmax()
    
    rfm_df['is_high_risk'] = np.where(rfm_df['Cluster'] == high_risk_cluster_id, 1, 0)
    
    return rfm_df[['is_high_risk']]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data to create customer-level features.
    """
    # Ensure datetime type
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    customer_df = df.groupby('CustomerId').agg(
        total_transactions=('TransactionId', 'count'),
        total_value=('Value', 'sum'),
        avg_value=('Value', 'mean'),
        std_value=('Value', 'std'),
        unique_products=('ProductId', 'nunique'),
        most_frequent_channel=('ChannelId', lambda x: x.mode()[0])
    ).reset_index()

    # Fill NaNs in std_dev (for customers with only 1 transaction, std is NaN)
    customer_df['std_value'].fillna(0, inplace=True)
    return customer_df

def get_preprocessor() -> ColumnTransformer:
    """Returns a scikit-learn ColumnTransformer for preprocessing the feature set."""
    numerical_features = ['total_transactions', 'total_value', 'avg_value', 'std_value', 'unique_products']
    categorical_features = ['most_frequent_channel']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor