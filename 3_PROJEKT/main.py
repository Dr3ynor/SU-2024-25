import numpy as np

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
    # Example dataset
    X = np.array([
        [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80], [24, 79]
    ])

    dbscan = DBSCAN(eps=2, min_samples=2)
    dbscan.fit(X)

    print("Cluster labels:", dbscan.labels)
