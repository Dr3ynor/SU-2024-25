import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = []

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
            
            if self.labels[neighbor_idx] == -1:  # Change noise to cluster point
                self.labels[neighbor_idx] = cluster_id
            
            elif self.labels[neighbor_idx] == -1 or self.labels[neighbor_idx] == -1:  # Process unvisited points
                self.labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            i += 1

if __name__ == "__main__":
    data = pd.read_csv("iris.csv")

    # Example preprocessing: select numerical columns and handle missing values
    numerical_data = data.select_dtypes(include=[np.number]).fillna(0)

    # Normalize data (optional, improves performance)
    normalized_data = normalize(numerical_data)


    # Convert to numpy array for DBSCAN
    X = normalized_data

    best_eps = None
    best_min_samples = None
    best_score = -1

    eps_values = np.linspace(0.1, 0.5, 20)
    print(eps_values)
    min_samples_values = range(2, 40)

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit(X)
            print(f"eps: {eps}, min_samples: {min_samples}, Number of clusters found: {len(set(labels.labels)) - (1 if -1 in labels.labels else 0)}")
            # Ignore clusters with all noise
            if len(set(labels.labels)) == 1:
                continue
            else:
                score = silhouette_score(X, labels.labels)
                print(f"Silhouette Score: {score}")
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples


    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best Silhouette Score: {best_score}")
    # Best eps: 0.4789473684210527, Best min_samples: 26, Best Silhouette Score: 0.7005873260504378
    # 5, 25
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)  # Adjust eps and min_samples as needed
    dbscan.fit(X)

    print("Cluster labels:", dbscan.labels)

    # Count the number of clusters (excluding noise)
    n_clusters = len(set(dbscan.labels))
    print("Number of clusters found (with noise cluster):", n_clusters)

    # Plotting
    plt.scatter(X[:,0], X[:,1], c=dbscan.labels, cmap='viridis', s=50, alpha=0.5)
    plt.show()


