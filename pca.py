import numpy as np


def PCA(x, new_dim):
    """
    PCA dimensionality reduction
    :param x: Input samples
    :param new_dim: Input samples new dimensions after performing dimensionality reduction
    :return: Numpy array containing the reduced-dimension input samples
    """
    # Centering the data
    x = x - np.mean(x, axis=0)

    # Covariance matrix calculation
    covariance_matrix = np.cov(x, rowvar=False)

    # Calculating the eigenvalues and eigenvectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    # Sorting the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]

    # Sorting the eigenvectors in descending order
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Selecting the top V eigenvectors, where V = new_dim
    eigenvector_subset = sorted_eigenvectors[:, 0:new_dim]

    # Transforming the original data
    x_reduced = np.dot(eigenvector_subset.transpose(), x.transpose()).transpose()

    return x_reduced
