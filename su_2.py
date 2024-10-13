import sklearn.cluster
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

"""
plt.rcParams.update({'figure.max_open_warning': 0})

for data_file in ['clusters3', 'clusters5', 'clusters5n', 'annulus', 'boxes', 'densegrid']:
    X = np.loadtxt('data_clustering/{}.csv'.format(data_file), delimiter=';')

    for n_cluster in [2, 3, 5]:
        for metric in ['euclidean', 'cityblock']:
            for linkage in ['single', 'complete']:
                key = f'{data_file}_{n_cluster}_{metric}_{linkage}'
                clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage).fit(X)
                plt.figure()
                plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
                plt.title(key)
                # plt.show()
                plt.savefig(f'output/{key}.png')
"""

def custom_agglomerative_clustering(X, n_clusters, metric, linkage):
    # Compute the distance matrix
    dist_matrix = squareform(pdist(X, metric=metric))
    np.fill_diagonal(dist_matrix, np.inf)
    
    clusters = [[i] for i in range(len(X))]
    
    while len(clusters) > n_clusters:
        # Find the closest pair of clusters
        min_dist = np.inf
        to_merge = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if linkage == 'single':
                    dist = np.min([dist_matrix[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                elif linkage == 'complete':
                    dist = np.max([dist_matrix[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)
        
        # Merge the closest pair of clusters
        clusters[to_merge[0]].extend(clusters[to_merge[1]])
        del clusters[to_merge[1]]
    
    # Assign cluster labels
    labels = np.zeros(len(X), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id
    
    return labels

for data_file in ['clusters3', 'clusters5', 'clusters5n', 'annulus', 'boxes', 'densegrid']:
    X = np.loadtxt('data_clustering/{}.csv'.format(data_file), delimiter=';')

    for n_cluster in [2, 3, 5]:
        for metric in ['euclidean', 'cityblock']:
            for linkage in ['single', 'complete']:
                key = f'{data_file}_{n_cluster}_{metric}_{linkage}'
                labels = custom_agglomerative_clustering(X, n_cluster, metric, linkage)
                plt.figure()
                plt.scatter(X[:, 0], X[:, 1], c=labels)
                plt.title(key)
                plt.savefig(f'my_output/{key}_custom.png')
