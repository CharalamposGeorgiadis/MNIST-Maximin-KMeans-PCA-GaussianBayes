import numpy as np
from numba import jit
from euclideanDistance import EuclideanDistance


# This command is utilized by the Numba library in order to speed up the code
@jit(nopython=True)
def Maximin(number_of_centers, x, n_samples):
    """
    Maximin algorithm for calculating the initial cluster of the k-means clustering algorithm

    :param number_of_centers: Number of clusters
    :param x: Input samples
    :param n_samples: Number of input samples
    :return: A numpy array containing the initial centers for the k-means Clustering algorithm
    """

    n_dimensions = x.shape[1]
    # Step 1: c1 = random x
    centers = np.zeros((number_of_centers, n_dimensions))
    centers[0] = x[0]

    # Step 2: Finding c2
    distances = np.zeros((number_of_centers, n_samples))
    distances[0][0] = EuclideanDistance(x[0], centers[0])
    max_distance = distances[0][0]
    max_x = 0
    for i in range(1, n_samples):
        distances[0][i] = EuclideanDistance(x[i], centers[0])
        if distances[0][i] > max_distance:
            max_distance = distances[0][i]
            max_x = i
    centers[1] = x[max_x]

    # Step 3: Calculating Max(Δi)
    current_centers = 2
    current_calculated_distances = 1
    while current_centers < 4:  # Stopping criterion
        max_distance = distances[0][0]
        max_x = 0
        for i in range(n_samples):
            min_distance = distances[0][i]
            # Calculating the distances between each sample and the newest center
            for j in range(current_calculated_distances, current_centers):
                distances[j][i] = EuclideanDistance(x[i], centers[j])
            # Finding the closest center to each sample
            for j in range(current_centers):
                if distances[j][i] < min_distance:
                    min_distance = distances[j][i]
            # Finding the maximum minimum distance
            if min_distance > max_distance:
                max_distance = min_distance
                max_x = i
        centers[current_centers] = x[max_x]
        current_centers += 1
        current_calculated_distances += 1
    return centers
