import numpy as np
from loadDataset import LoadDataset
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate
from featureVector import FeatureVector
from maximin import Maximin
from kMeansClustering import KMeansClustering
from pca import PCA
from gaussianBayesClassifier import GaussianBayesClassifier

# Loading the MNIST dataset
x_train, y_train, x_test, y_test = LoadDataset()

# Reshaping the training and test images into 28x28 matrices
x_train_reshaped = x_train.reshape(-1, 28, 28)

# All possible training and test sample labels
real_labels = np.array([1, 3, 7, 9])

# Specifying Legend colors
legend = [Line2D([0], [0], marker='o', color="w", label='y=1', markerfacecolor="red", markersize=10),
          Line2D([0], [0], marker='o', color="w", label='y=3', markerfacecolor="green", markersize=10),
          Line2D([0], [0], marker='o', color="w", label='y=7', markerfacecolor="blue", markersize=10),
          Line2D([0], [0], marker='o', color="w", label='y=9', markerfacecolor="yellow", markersize=10)]

# Calculating the 2D feature vector of the training set
print("Calculating the training image Feature Vectors...")
n_samples = x_train_reshaped.shape[0]
train_feature_vectors = FeatureVector(x_train_reshaped, n_samples)

# Colormap for the scatter plot of the 2D Feature Vector
colormap = ['red' if y == 1 else 'green' if y == 3 else 'blue' if y == 7 else 'yellow' for y in y_train]

# Plotting the scatter plot of the 2D Feature Vectors
plt.scatter(train_feature_vectors[:, 0], train_feature_vectors[:, 1], color=colormap)
plt.gca().set(xlim=(0, 0.3), ylim=(0, 0.35),
              xlabel='First Feature Vector Component', ylabel='Second Feature Vector Component')
plt.title("2D Feature Vector of Each Image")
plt.legend(handles=legend)
plt.show()

# Calculating k-means Clustering centroids of the training Feature Vector using the Maximin algorithm
print("Calculating k-means Clustering centroids using Maximin...\n")
# Calculating the centers for the k-means Clustering algorithm using Maximin
k_means_centers = Maximin(4, train_feature_vectors, n_samples)

# k-means Clustering
k_means_labels, centroids = KMeansClustering(k_means_centers, train_feature_vectors, y_train, n_samples, real_labels)

# Colormap for the scatter plot of the 2D Feature Vectors after k-means Clustering
colormap = ['red' if y == 1 else 'green' if y == 3 else 'blue' if y == 7 else 'yellow' for y in k_means_labels]

# Plotting the scatter plot of the 2D Feature Vectors after k-means Clustering
plt.scatter(train_feature_vectors[:, 0], train_feature_vectors[:, 1], color=colormap)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, label='Centroids', color='black')
plt.gca().set(xlim=(0, 0.3), ylim=(0, 0.35),
              xlabel='First Feature Vector Component', ylabel='Second Feature Vector Component')
plt.title("Clustered 2D Feature Vector of Each Image")
legend.append(Line2D([0], [0], marker='o', color="w", label='Centroids', markerfacecolor="black", markersize=10))
plt.legend(handles=legend)
plt.show()

# Calculating the purity of the k-means Clustering algorithm
k_means_purity = np.sum(k_means_labels == y_train) / len(y_train)
print(f"K-Means Clustering Purity: {k_means_purity * 100:.2f}%\n")

# PCA dimensionality reduction
pca_purities = np.zeros(4)

# PCA for V = 2
print("Performing PCA for V = 2...")
x_train_pca_2 = PCA(x_train, 2)

# Plotting the scatter plot of the 2D training images
plt.scatter(x_train_pca_2[:, 0], x_train_pca_2[:, 1], color=colormap)
plt.gca().set(xlim=(-8.5, 5), ylim=(-8, 4),
              xlabel='First Principal Component', ylabel='Second Principal Component')
plt.title("2D Training Samples after PCA")
legend.pop()
plt.legend(handles=legend)
plt.show()

# Calculating k-means Clustering centroids of the reduced-dimension training set using the Maximin algorithm
k_means_centers = Maximin(4, x_train_pca_2, n_samples)

# k-means Clustering
k_means_labels, centroids = KMeansClustering(k_means_centers, x_train_pca_2, y_train, n_samples, real_labels)

# Colormap for the scatter plot of the 2D training images after k-means Clustering
colormap = ['red' if y == 1 else 'green' if y == 3 else 'blue' if y == 7 else 'yellow' for y in k_means_labels]

# Plotting the scatter plot of the 2D training images after k-means Clustering
plt.scatter(x_train_pca_2[:, 0], x_train_pca_2[:, 1], color=colormap)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, label='Centroids', color='black')
plt.gca().set(xlim=(-8, 5), ylim=(-8, 4),
              xlabel='First Principal Component', ylabel='Second Principal Component')
plt.title("Clustered Training Samples for PCA=2")
legend.append(Line2D([0], [0], marker='o', color="w", label='Centroids', markerfacecolor="black", markersize=10))
plt.legend(handles=legend)
plt.show()

# Calculating the purity of the k-means Clustering algorithm
k_means_purity = np.sum(k_means_labels == y_train) / len(y_train)
pca_purities[0] = round(k_means_purity * 100, 2)

# PCA for V = 25
print("Performing PCA for V = 25...")
x_train_pca_25 = PCA(x_train, 25)

# Calculating k-means Clustering centroids of the 25D training images using the Maximin algorithm
k_means_centers = Maximin(4, x_train_pca_25, n_samples)
# k-means Clustering
k_means_labels, _ = KMeansClustering(k_means_centers, x_train_pca_25, y_train, n_samples, real_labels)

# Calculating the purity of the k-means Clustering algorithm
k_means_purity = np.sum(k_means_labels == y_train) / len(y_train)
pca_purities[1] = round(k_means_purity * 100, 2)

# PCA for V = 50
print("Performing PCA for V = 50...")
x_train_pca_50 = PCA(x_train, 50)

# Calculating k-means Clustering centroids of the 50D training images using the Maximin algorithm
k_means_centers = Maximin(4, x_train_pca_50, n_samples)
# k-means Clustering
k_means_labels, _ = KMeansClustering(k_means_centers, x_train_pca_50, y_train, n_samples, real_labels)

# Calculating the purity of the k-means Clustering algorithm
k_means_purity = np.sum(k_means_labels == y_train) / len(y_train)
pca_purities[2] = round(k_means_purity * 100, 2)

# PCA for V = 100
print("Performing PCA for V = 100...\n")
x_train_pca_100 = PCA(x_train, 100)

# Calculating k-means Clustering centroids of the 100D training images using the Maximin algorithm
k_means_centers = Maximin(4, x_train_pca_100, n_samples)
# k-means Clustering
k_means_labels, _ = KMeansClustering(k_means_centers, x_train_pca_100, y_train, n_samples, real_labels)

# Calculating the purity of the k-means Clustering algorithm
k_means_purity = np.sum(k_means_labels == y_train) / len(y_train)
pca_purities[3] = round(k_means_purity * 100, 2)

# Printing the k-means Clustering Purities for each chosen PCA dimension
print(tabulate([['V=2', str(pca_purities[0]) + "%"],
                ['V=25', str(pca_purities[1]) + "%"],
                ['V=50', str(pca_purities[2]) + "%"],
                ['V=100', str(pca_purities[3]) + "%"]],
               headers=['PCA Dimensions', 'Clustering Purity']))

# Finding the value of V that reached the highest clustering purity
tested_V = [2, 25, 50, 100]
best_purity = np.argmax(pca_purities)
if best_purity == 0:
    x_train_pca = x_train_pca_2
elif best_purity == 1:
    x_train_pca = x_train_pca_25
elif best_purity == 2:
    x_train_pca = x_train_pca_50
elif best_purity == 3:
    x_train_pca = x_train_pca_100
print("\nBest PCA purity is for V = " + str(tested_V[best_purity]))

# Gaussian Naive Bayes Classifier for the best V
print("\nGaussian Naive Bayes Classifier training and evaluation for PCA=" + str(tested_V[best_purity]) + " ...")

# Training set evaluation
predictions = GaussianBayesClassifier(x_train_pca, y_train, x_train_pca, 4, real_labels)
accuracy = np.sum(predictions == y_train) / len(y_train)
print(f"Gaussian Naive Bayes accuracy on the training set: {accuracy * 100:.2f}%")

# Test set evaluation
x_test_pca = PCA(x_test, tested_V[best_purity])
predictions = GaussianBayesClassifier(x_train_pca, y_train, x_test_pca, 4, real_labels)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Gaussian Naive Bayes accuracy on the test set: {accuracy * 100:.2f}%")
