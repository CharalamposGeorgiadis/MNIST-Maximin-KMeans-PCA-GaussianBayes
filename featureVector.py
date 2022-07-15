import numpy as np
from numba import jit


# This command is utilized by the Numba library in order to speed up the code
@jit(nopython=True)
def FeatureVector(train_data, iterations):
    """
    Calculates the 2D Feature Vector of each image of the training set.

    :param train_data: Training samples
    :param iterations: Number of training samples
    :return A numpy array containing a 2D feature vector of each image of the training set
    """
    feature_vector = np.zeros((train_data.shape[0], 2))
    for i in range(iterations):
        first_feature = 0
        count = 0
        # Calculating the first feature
        for j in range(1, 28, 2):
            first_feature += np.mean(train_data[i][j])
            count += 1
        first_feature /= count
        second_feature = 0
        count = 0
        # Calculating the second feature
        for j in range(0, 28, 2):
            second_feature += np.mean(train_data[i][:][j])
            count += 1
        second_feature /= count
        feature_vector[i][0] = first_feature
        feature_vector[i][1] = second_feature
    return feature_vector
