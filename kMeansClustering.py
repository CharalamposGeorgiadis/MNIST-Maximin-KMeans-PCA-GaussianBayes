import numpy as np
from numba import jit
from euclideanDistance import EuclideanDistance


# This command is utilized by the Numba library in order to speed up the code
@jit(nopython=True)
def KMeansClustering(centers, x, y, n_samples, class_labels):
    """
    K-Means Clustering

    :param class_labels: Label of each cluster. These labels do not correspond to real class labels
    :param centers: Initial centers
    :param x: Input samples
    :param y: Input sample labels
    :param n_samples: Number of input samples
    :param class_labels: All possible input sample labels
    :return: Numpy array containing the new labels of the input samples, cluster centroids
    """
    n_dimensions = x.shape[1]
    number_of_centers = centers.shape[0]
    cluster_labels = np.zeros(n_samples, dtype=np.uint64)

    # Step 2: Finding the labels of each sample
    while True:
        stop = True  # Stopping criterion
        for i in range(n_samples):
            min_distance = EuclideanDistance(x[i], centers[0])
            cluster_labels[i] = 0
            for j in range(1, number_of_centers):
                distance = EuclideanDistance(x[i], centers[j])
                if distance < min_distance:
                    min_distance = distance
                    cluster_labels[i] = j

        # Step 3: Updating centers
        for i in range(number_of_centers):
            sum = np.zeros(n_dimensions)
            count = 0
            for j in range(n_samples):
                if cluster_labels[j] == i:
                    sum += x[j]
                    count += 1
            new_center = sum / count
            if not np.array_equal(new_center, centers[i]):
                stop = False
                centers[i] = new_center

        # Step 4: Checking Stopping Criteria
        if stop:
            break

    # Converting the cluster labels to real labels
    max = np.zeros(len(centers), dtype=np.uint64)
    sorted_count_indices = np.zeros((len(centers), len(class_labels)))
    sorted_counts = np.zeros((len(centers), len(class_labels)))
    for i in range(len(centers)):
        count = np.zeros(len(centers))
        for j in range(len(y)):
            if cluster_labels[j] == i:
                for k in range(len(centers)):
                    if y[j] == class_labels[k]:
                        count[k] += 1
        sorted_count_indices[i] = np.argsort(count)
        sorted_counts[i] = np.sort(count)
        max[i] = sorted_count_indices[i][-1]

    for i in range(len(centers) - 1):
        for j in range(i + 1, len(max)):
            if max[i] == max[j]:
                if sorted_counts[i][-1] - sorted_counts[j][-1] < sorted_counts[i][-2] - sorted_counts[j][-2]:
                    max[i] = sorted_count_indices[i][-2]
                else:
                    max[j] = sorted_count_indices[i][-2]

    real_labels = np.zeros(len(y))
    for i in range(len(y)):
        real_labels[i] = class_labels[max[cluster_labels[i]]]
    return real_labels, centers
