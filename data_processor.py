"""Data processing module for customer segmentation."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data loading, cleaning, and preprocessing."""

    def __init__(self):
        """Initialize DataProcessor."""
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(filepath)
            logger.info(f"Data loaded: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def explore_data(self) -> dict:
        """Get basic data exploration information."""
        if self.data is None:
            raise ValueError("Data not loaded")

        return {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "basic_stats": self.data.describe().to_dict(),
        }

    def prepare_features(self, feature_indices: list = None) -> np.ndarray:
        """Prepare features for clustering."""
        if self.data is None:
            raise ValueError("Data not loaded")

        if feature_indices is None:
            features = self.data.iloc[:, [3, 4]].values
        else:
            features = self.data.iloc[:, feature_indices].values

        self.processed_data = features
        return features

    def get_feature_statistics(self) -> dict:
        """Get statistics for the processed features."""
        if self.processed_data is None:
            raise ValueError("Features not prepared")

        return {
            "mean_income": float(np.mean(self.processed_data[:, 0])),
            "std_income": float(np.std(self.processed_data[:, 0])),
            "mean_spending": float(np.mean(self.processed_data[:, 1])),
            "std_spending": float(np.std(self.processed_data[:, 1])),
            "min_income": float(np.min(self.processed_data[:, 0])),
            "max_income": float(np.max(self.processed_data[:, 0])),
            "min_spending": float(np.min(self.processed_data[:, 1])),
            "max_spending": float(np.max(self.processed_data[:, 1])),
        }
