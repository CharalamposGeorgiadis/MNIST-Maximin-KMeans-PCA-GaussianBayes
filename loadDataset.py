import numpy as np
import pandas as pd


# Function that loads the dataset
# return: Training samples, labels of the training samples, test samples, labels of the test samples
def LoadDataset():
    # Loading the training set
    train_set = pd.read_csv('dataset/mnist_train.csv')

    # Keeping only the entries for labels 1, 3, 7, 9
    train_set = train_set[- train_set['label'].isin([0, 2, 4, 5, 6, 8])]

    # Separating the training set images and their labels
    x_train = train_set.drop(columns=['label'])
    # Normalizing the training samples
    x_train = x_train.astype(np.float64) / 255
    y_train = train_set['label'].values

    # Loading the test set
    test_set = pd.read_csv('dataset/mnist_test.csv')

    # Keeping only the entries for labels 1, 3, 7, 9
    test_set = test_set[- test_set['label'].isin([0, 2, 5, 4, 6, 8])]

    # Separating the test set images and their labels
    x_test = test_set.drop(columns=['label'])
    # Normalizing the test samples
    x_test = x_test.astype(np.float64) / 255
    y_test = test_set['label'].values

    return np.array(x_train), y_train, np.array(x_test), y_test
