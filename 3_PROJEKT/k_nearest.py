import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import Counter

class KNearestNeighbor:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Ukládá trénovací data a jejich labely.
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """
        Predikuje labely pro vstupní data.
        """
        predictions = []
        for point in X:
            distances = self._compute_distances(point)
            neighbors = self._get_k_nearest_neighbors(distances)
            most_common_label = self._majority_vote(neighbors)
            predictions.append(most_common_label)
        return predictions
    
    def _compute_distances(self, point):
        """
        Počítá vzdálenosti mezi daným bodem a všemi trénovacími daty.
        """
        distances = np.linalg.norm(self.X_train - point, axis=1)
        return distances
    
    def _get_k_nearest_neighbors(self, distances):
        """
        Vrací indexy k nejbližších sousedů.
        """
        neighbor_indices = np.argsort(distances)[:self.k]
        return neighbor_indices
    
    def _majority_vote(self, neighbor_indices):
        """
        Vrací nejčastější label mezi k nejbližšími sousedy.
        """
        neighbor_labels = self.y_train[neighbor_indices]
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]




data = pd.read_csv("iris.csv")
print(data.head())
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# Example preprocessing: select numerical columns and handle missing values
numerical_data = data.select_dtypes(include=[np.number]).fillna(0)

# Normalize data (optional, improves performance)
normalized_data = normalize(numerical_data)

# Convert to numpy array for DBSCAN
X = normalized_data

best_k = None
best_silhouette = -1
y=data['species']

for k in range(2, 11):
    model = KNearestNeighbor(k)
    model.fit(X, y)
    silhouette = silhouette_score(X, model.predict(X))
    print(f"Silhouette score for k={k}: {silhouette}")
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k

print(f"Best k: {best_k}, Best silhouette: {best_silhouette}")

# Plot the best clustering
model = KNearestNeighbor(best_k)
model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Best clustering with k={best_k} (custom implementation)")
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

best_k = None
best_silhouette = -1

for k in range(2, 11):
    model = KNeighborsClassifier(k)
    model.fit(X, y)
    silhouette = silhouette_score(X, model.predict(X))
    print(f"Silhouette score for k={k}: {silhouette}")
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k

print(f"Best k: {best_k}, Best silhouette: {best_silhouette}")


# Plot the best clustering
model = KNeighborsClassifier(best_k)
model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X))
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.title(f"Best clustering with k={best_k} (sklearn)")
plt.show()