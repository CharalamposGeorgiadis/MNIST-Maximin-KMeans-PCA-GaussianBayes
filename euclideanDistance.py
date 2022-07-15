import numpy as np
from numba import jit


# This command is utilized by the Numba library in order to speed up the code
@jit(nopython=True)
# Function that calculates the euclidean distance between two inputs
# param x1: first input
# param x2: second input
def EuclideanDistance(x1, x2):
    return np.sqrt(np.sum(np.square((x1 - x2))))
