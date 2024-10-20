import numpy as np
import matplotlib.pyplot as plt

# vytvoř 2 matice - manhattan a euclidean, udělat pro ně single a complete linkage pro obě


def plot_clusters(data, clusters, title, save_state):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='rainbow')
    plt.title(title)
    if save_state.lower() == 'y':
        plt.savefig(title + '.png')
    plt.show()

def find_clusters(linkage_matrix, num_clusters):
    clusters = {i: [i] for i in range(len(linkage_matrix) + 1)}
    for i, (c1, c2, _, _) in enumerate(linkage_matrix):
        if len(clusters) <= num_clusters:
            break
        new_cluster = clusters.pop(int(c1)) + clusters.pop(int(c2))
        clusters[len(linkage_matrix) + i + 1] = new_cluster
    labels = np.zeros(len(linkage_matrix) + 1, dtype=int)
    for cluster_id, members in clusters.items():
        for member in members:
            labels[member] = cluster_id
    return labels

def agglomerative_clustering(dist_matrix, linkage_method, num_clusters):
    n = dist_matrix.shape[0]
    clusters = {i: [i] for i in range(n)}
    linkage_matrix = []

    while len(clusters) > 1:
        min_dist = float('inf')
        to_merge = None

        for i, c1 in clusters.items():
            for j, c2 in clusters.items():
                if i >= j:
                    continue
                if linkage_method == 'single':
                    dist = np.min([dist_matrix[p1, p2] for p1 in c1 for p2 in c2])
                elif linkage_method == 'complete':
                    dist = np.max([dist_matrix[p1, p2] for p1 in c1 for p2 in c2])
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)

        if to_merge is None:
            break

        i, j = to_merge
        new_cluster = clusters.pop(i) + clusters.pop(j)
        new_cluster_id = n + len(linkage_matrix)
        clusters[new_cluster_id] = new_cluster
        linkage_matrix.append([i, j, min_dist, len(new_cluster)])

    linkage_matrix = np.array(linkage_matrix)
    return find_clusters(linkage_matrix, num_clusters)


def normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def standardize(data):
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    standardized_data = (data - mean_vals) / std_vals
    return standardized_data


def main():
    filename = input("Filename:")
    num_clusters = int(input("Number of clusters:"))
    save_state = input("Do you want to save the image? (y/n):")
    data = np.loadtxt(filename, delimiter=';')
    normalized_data= normalize(data)
    standardized_data = standardize(data)

    manhattan_dist_matrix = np.abs(data[:, np.newaxis] - data).sum(axis=2)
    euclidean_dist_matrix = np.sqrt(((data[:, np.newaxis] - data) ** 2).sum(axis=2))
    cosine_dist_matrix = 1 - np.dot(data, data.T) / (np.linalg.norm(data, axis=1)[:, np.newaxis] * np.linalg.norm(data, axis=1))

    # plot_clusters(data, np.zeros(data.shape[0]), "Original Data",save_state)

    #cosine_clusters_single_normalized = agglomerative_clustering(cosine_dist_matrix, 'single', num_clusters)
    #plot_clusters(normalized_data, cosine_clusters_single_normalized, "Normalized Cosine Distance - Single Linkage",save_state)

    #cosine_clusters_single_standardized = agglomerative_clustering(cosine_dist_matrix, 'single', num_clusters)
    #plot_clusters(standardized_data, cosine_clusters_single_standardized, "Standardized Cosine Distance - Single Linkage",save_state)


    #manhattan_clusters_single = agglomerative_clustering(manhattan_dist_matrix, 'single', num_clusters)
    #plot_clusters(data, manhattan_clusters_single, "Manhattan Distance - Single Linkage",save_state)

    #manhattan_clusters_complete = agglomerative_clustering(manhattan_dist_matrix, 'complete', num_clusters)
    #plot_clusters(data, manhattan_clusters_complete, "Manhattan Distance - Complete Linkage",save_state)

    euclidean_clusters_single = agglomerative_clustering(euclidean_dist_matrix, 'single', num_clusters)
    plot_clusters(data, euclidean_clusters_single, "Euclidean Distance - Single Linkage")
    
    euclidean_clusters_complete = agglomerative_clustering(euclidean_dist_matrix, 'complete', num_clusters)
    plot_clusters(data, euclidean_clusters_complete, "Euclidean Distance - Complete Linkage",save_state)


if __name__ == '__main__':
    main()