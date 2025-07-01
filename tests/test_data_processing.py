import pandas as pd
import pytest
from src.data_processing import create_features, create_proxy_target

# Sample data for testing
@pytest.fixture
def sample_raw_data():
    """Provides a sample DataFrame for testing."""
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'Value': [100, 200, 50, 1000],
        'ProductId': ['P1', 'P2', 'P1', 'P3'],
        'ChannelId': ['CH1', 'CH1', 'CH2', 'CH1'],
        'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-10', '2023-01-05', '2022-12-01'])
    }
    return pd.DataFrame(data)

def test_create_features_output_shape(sample_raw_data):
    """Tests if the output DataFrame has one row per unique customer."""
    features_df = create_features(sample_raw_data)
    # 3 unique customers: C1, C2, C3
    assert features_df.shape[0] == 3
    assert 'CustomerId' in features_df.columns

def test_create_features_std_value_handling(sample_raw_data):
    """Tests if std_value is correctly calculated and NaN is filled for single transactions."""
    features_df = create_features(sample_raw_data)
    
    # For customer C2 (single transaction), std_value should be 0
    std_c2 = features_df[features_df['CustomerId'] == 'C2']['std_value'].iloc[0]
    assert std_c2 == 0.0
    
    # For customer C1 (multiple transactions), std_value should be non-zero
    std_c1 = features_df[features_df['CustomerId'] == 'C1']['std_value'].iloc[0]
    assert std_c1 > 0.0