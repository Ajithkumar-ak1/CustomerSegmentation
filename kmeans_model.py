"""K-means clustering model for customer segmentation."""

import numpy as np
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class KMeansSegmentation:
    """K-means clustering model for customer segmentation."""

    def __init__(self, random_state: int = 42):
        """Initialize KMeansSegmentation."""
        self.model = None
        self.random_state = random_state
        self.wcss = []
        self.k_values = []

    def find_optimal_clusters(self, features: np.ndarray, k_range: tuple = (1, 11)):
        """Find optimal clusters using elbow method."""
        self.wcss = []
        self.k_values = list(range(k_range[0], k_range[1]))

        for k in self.k_values:
            kmeans = KMeans(
                n_clusters=k, init="k-means++", random_state=self.random_state
            )
            kmeans.fit(features)
            self.wcss.append(kmeans.inertia_)

        return self.k_values, self.wcss

    def train(self, features: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Train K-means model."""
        self.model = KMeans(
            n_clusters=n_clusters, init="k-means++", random_state=self.random_state
        )
        labels = self.model.fit_predict(features)
        logger.info(f"Model trained with {n_clusters} clusters")
        return labels

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(features)

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.cluster_centers_

    def get_cluster_info(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """Get information about each cluster."""
        cluster_info = {}
        for cluster in np.unique(labels):
            cluster_mask = labels == cluster
            cluster_features = features[cluster_mask]

            cluster_info[f"Cluster {cluster}"] = {
                "Size": int(np.sum(cluster_mask)),
                "Percentage": f"{np.sum(cluster_mask) / len(labels) * 100:.1f}%",
                "Avg Income": f"${np.mean(cluster_features[:, 0]):.0f}k",
                "Avg Spending": f"{np.mean(cluster_features[:, 1]):.0f}",
            }

        return cluster_info
