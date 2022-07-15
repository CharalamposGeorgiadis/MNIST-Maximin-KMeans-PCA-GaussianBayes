import numpy as np
from numba import jit


def GaussianBayesClassifier(x1, y, x2, n_classes, labels):
    """
    Gaussian Naive Bayes Classifier training and predictions
    :param x1: Training samples
    :param y: Training sample labels
    :param x2: Test samples
    :param n_classes: Number of classes
    :param labels: Class labels
    :return: Numpy array containing the predictions for the test samples
    """

    # Prior probabilities for each class
    priors = np.zeros(n_classes)

    # Means and variances for each feature for each class
    means = np.zeros((n_classes, x1.shape[1]))
    variances = np.zeros((n_classes, x1.shape[1]))

    # Test sample predictions
    pred = np.zeros(x2.shape[0], dtype=int)

    # Calculating the prior probabilities of each class and the means and variances for each feature of each class
    for i in range(n_classes):
        current_class_samples = x1[y == labels[i]]
        means[i] = np.mean(current_class_samples, axis=0)
        variances[i] = np.var(current_class_samples, axis=0)
        priors[i] = len(current_class_samples) / len(x1)

    # Calculating the Posterior probabilities for each test sample
    for i in range(x2.shape[0]):
        posteriors = np.copy(priors)
        for j in range(n_classes):
            for k in range(x2.shape[1]):
                posteriors[j] *= GaussianPDF(x2[i][k], means[j][k], variances[j][k])
        pred[i] = np.argmax(posteriors)
    for i in range(len(pred)):
        pred[i] = labels[pred[i]]
    return pred


# This command is utilized by the Numba library in order to speed up the code
@jit(nopython=True)
def GaussianPDF(feature, mean, variance):
    """
    Gaussian PDF calculation
    :param feature: Input feature
    :param mean: Mean of the input feature for one class
    :param variance: Variance of the input feature for one class
    :return: Float containing the Gaussian PDF
    """
    exp = np.exp((-np.power((feature - mean), 2)) / (2 * variance))
    fraction = np.sqrt(2 * np.pi * variance)
    return exp / fraction
