import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

# Define function for loading data
def load_data(subset: int = 250):
    """Function for loading data
    Args:
        subset (int): Integer specifying size of subset
        
    Returns:
        Train, val and test data sets with labels
    """
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Create validation set
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, train_size=0.5, random_state=42
    )

    # Subset for faster processing
    X_train = X_train[: subset * 2]
    y_train = y_test[: subset * 2]
    X_val = X_val[:subset]
    y_val = y_val[:subset]
    X_test = X_test[:subset]
    y_test = y_test[:subset]

    return (X_train, X_val, X_test, y_train, y_val, y_test,) 


# Define W(d, l) - Function that calculates impending corrosion speed - also based on y.
def Q(d, l):
    """Function for determining rate of corrosion

    Args:
        d (int): Difference in grayness
        l (float): scaling factor
    """
    return (255 - d) * l


# Function for defining context kernel
def context(seed: np.ndarray, r: int, c: int):
    """Defines a list of all cells neigbouring a given cell

    Args:
        seed (np.ndarray): 2D array of image
        r (int): Row number of the given cell
        c (int): Collumn number of the given cell

    Returns:
        List: List of values for the 9 cells
    """
    return [
        seed[r, c],
        seed[r, c + 1],
        seed[r, c + 2],
        seed[r + 1, c],
        seed[r + 1, c + 2],
        seed[r + 2, c],
        seed[r + 2, c + 1],
        seed[r + 2, c + 2],
    ]


# Define function for binarising MNIST images
def binarise(img, threshold):
    """Binarizes a 2D image on the basis of a threshold (dead cells = 0, alive cells = 1)

    Args:
        img (np.ndarray): 2D array to binarize
        threshold (int): Threshold for binarizing upon

    Returns:
        np.ndarray: Binarized image
    """

    img[img >= threshold] = 255
    img[img < threshold] = 0

    return img
