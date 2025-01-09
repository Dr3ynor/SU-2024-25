import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.labels = [-1] * len(X)  # Initialize all points as noise (-1)
        cluster_id = 0

        for i in range(len(X)):
            if self.labels[i] != -1:
                continue  # Skip if already processed
            
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Mark as noise
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

        return self

    def _region_query(self, X, point_idx):
        neighbors = []
        for i in range(len(X)):
            if np.linalg.norm(X[point_idx] - X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id  # Change noise to border point
            
            if self.labels[neighbor_idx] == -1 or self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)

                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            i += 1

if __name__ == "__main__":
    # Load Titanic dataset
    data = pd.read_csv("titanic.csv")

    # Example preprocessing: select numerical columns and handle missing values
    numerical_data = data.select_dtypes(include=[np.number]).fillna(0)

    # Normalize data (optional, improves performance)
    normalized_data = normalize(numerical_data)


    # Convert to numpy array for DBSCAN
    X = normalized_data

    dbscan = DBSCAN(eps=0.3, min_samples=3)  # Adjust eps and min_samples as needed
    dbscan.fit(X)

    print("Cluster labels:", dbscan.labels)

    # Count the number of clusters (excluding noise)
    n_clusters = len(set(dbscan.labels)) - (1 if -1 in dbscan.labels else 0)
    print("Number of clusters found:", n_clusters)
