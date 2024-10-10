import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
# 2 matice - manhattan a euclidean, udělat pro ně single a complete linkage pro obě

for data_file in ['clusters3', 'clusters5', 'clusters5n', 'annulus', 'boxes', 'densegrid']:
    data = np.loadtxt('/home/jakub/school/7th_semester/SU-2024-25/2_tyden/{}.csv'.format(data_file), delimiter=';')
# Compute distance matrices
manhattan_dist = squareform(pdist(data, metric='cityblock'))
euclidean_dist = squareform(pdist(data, metric='euclidean'))

# Perform agglomerative clustering
def agglomerative_clustering(dist_matrix, linkage_method, num_clusters):
    linkage_matrix = sch.linkage(dist_matrix, method=linkage_method)
    clusters = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    return linkage_matrix, clusters

# Plotting function
def plot_clusters(data, clusters, title):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='rainbow')
    plt.title(title)
    plt.show()

# Perform clustering and plot results
for dist_matrix, dist_name in [(manhattan_dist, 'Manhattan'), (euclidean_dist, 'Euclidean')]:
    for linkage_method in ['single', 'complete']:
        linkage_matrix, clusters = agglomerative_clustering(dist_matrix, linkage_method, num_clusters=3)
        plot_clusters(data, clusters, f'{dist_name} Distance - {linkage_method.capitalize()} Linkage')
        