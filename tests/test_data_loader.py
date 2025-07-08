import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from political_party_analysis.loader import DataLoader
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """
    Create a mock DataFrame for testing.
    It includes duplicates, NaN values, and non-feature columns.
    Returns:
        pd.DataFrame: The mock DataFrame.
    """
    data = {
        "id": [0, 1, 2, 2],
        "col1": [1, 2, 3, 3],
        "col2": [30.0, np.nan, 30.0, 30.0],
        "col3": [5, 7, 12, 12],
        "non_feature": ["a", "b", "c", "c"],
        "all_nans": [np.nan] * 4,
    }
    df = pd.DataFrame(data=data)
    return df

@pytest.fixture
def data_loader(mocker,mock_df: pd.DataFrame) -> DataLoader:
    """Fixture to create a DataLoader instance."""
    mocker.patch.object(DataLoader, "_download_data", return_value=mock_df)
    return DataLoader()


# --- Unit Tests ---

def test_data_loader_initialization(data_loader: DataLoader, mock_df: pd.DataFrame):
    """Test the initialization of the DataLoader.
    Args:
        data_loader (DataLoader): The DataLoader instance to test.
    """
    assert isinstance(data_loader, DataLoader)
    assert data_loader.party_data.shape == mock_df.shape

def test_remove_duplicates(data_loader: DataLoader, mock_df: pd.DataFrame):
    """
    Test the removal of duplicate rows from the DataFrame.
    """
    deduped_df = data_loader.remove_duplicates(mock_df)
    
    assert deduped_df.shape == (3, 6)
    assert deduped_df.index.tolist() == [0, 1, 2]

def test_remove_nonfeature_cols(data_loader: DataLoader, mock_df: pd.DataFrame):
    """
    Test the removal of non-feature columns and setting index.
    """
    processed_df = data_loader.remove_nonfeature_cols(mock_df, ["non_feature", "all_nans"], ["id"])
    
    assert processed_df.shape == (4, 3)
    assert processed_df.index.name == "id"
    assert "non_feature" not in processed_df.columns
    assert "all_nans" not in processed_df.columns

def test_handle_nan_values(data_loader: DataLoader, mock_df: pd.DataFrame):
    """Test the handling of NaN values in the DataFrame.
    """
    processed_df = data_loader.handle_NaN_values(mock_df)
    
    assert processed_df.isnull().sum().sum() == 0
    assert processed_df.shape == (4, 6)

def test_scale_features(data_loader: DataLoader, mock_df: pd.DataFrame):
    """
    Test a function to normalise values in a dataframe. Use StandardScaler.
    FIX: This test now correctly selects only numeric columns before scaling.
    """
    # StandardScaler can only operate on numeric data. Select numeric columns first.
    numeric_df = mock_df[["col1", "col3"]].drop_duplicates().reset_index(drop=True)
    
    scaled_df = data_loader.scale_features(numeric_df)

    scaler = StandardScaler()
    expected_scaled_array = scaler.fit_transform(numeric_df)
    expected_df = pd.DataFrame(expected_scaled_array, columns=numeric_df.columns)

    assert_frame_equal(scaled_df, expected_df)
    assert np.isclose(scaled_df["col1"].mean(), 0)
    assert np.isclose(scaled_df["col1"].std(ddof=0), 1, atol=0.01)


def test_preprocess_data(mocker, mock_df: pd.DataFrame):
    """Test the complete preprocessing pipeline of the DataLoader.
    Args:
        mocker: The pytest-mock fixture to mock methods.
        mock_df (pd.DataFrame): The mock DataFrame to use for testing.
    """
    data_loader = DataLoader()
    mocker.patch.object(data_loader, "party_data", mock_df)
    mocker.patch.object(data_loader, "non_features", ["non_feature","all_nans"])
    mocker.patch.object(data_loader, "index", ["id"])
    
    #run the entire pipeline
    data_loader.preprocess_data()
    
    
    expected_df = pd.DataFrame(
        data={
            "col1": [-1.225, 0, 1.225],
            "col2": [0.0] * 3,
            "col3": [-1.019, -0.340, 1.359],
        },
        index=[0, 1, 2],
    )
    expected_df.index.name = "id"
    assert_frame_equal(data_loader.party_data, expected_df, rtol=3)


# --- Integration Test ---

@pytest.mark.integration
def test_download_data():
    data_loader = DataLoader()
    assert data_loader.party_data.shape == (277, 55)
