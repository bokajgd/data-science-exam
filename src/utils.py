import numpy as np
import pandas as pd
import keras
import random
from keras import layers
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist
from keras import layers

# Define function for loading data
def load_data(dataset: str = "MNIST", subset: int = 250):
    """Function for loading data
    Args:
        subset (int): Integer specifying size of subset
        
    Returns:
        Train, val and test data sets with labels
    """
    # Load data
    if dataset == "MNIST":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if dataset == "Fashion MNIST":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Create validation set
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, train_size=0.5, random_state=42
    )

    # Subset for faster processing
    X_train = X_train[: subset * 2]
    y_train = y_train[: subset * 2]
    X_val = X_val[:subset]
    y_val = y_val[:subset]
    X_test = X_test[:subset]
    y_test = y_test[:subset]

    return (X_train, X_val, X_test, y_train, y_val, y_test) 


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
def binarise(img: np.array=None, threshold: int=0):
    """Binarizes a 2D image on the basis of a threshold (dead cells = 0, alive cells = 1)

    Args:
        img (np.ndarray): 2D array to binarize
        threshold (int): Threshold for binarizing upon

    Returns:
        np.ndarray: Binarized image
    """
    img = img.copy()
    img[img > threshold] = 255
    img[img <= threshold] = 0

    return img

# Define function for making 1D CNN model for timeseries classification
def make_1d_cnn_model(X_train_shape):
    """Takes the shape of X_train, and outputs a 1d CNN model for timeseries predictions

    Args:
        X_train_shape (tuple): Tuple with dimensions of the training data

    Returns:
        keras.models.Model: 1D CNN model for time series predictions
    """    
    # Define inputshape - should be (length of timeseries, 1)
    input_shape = (X_train_shape[1], 1)

    # Define architecture
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(10, activation="softmax")(gap)

    # Return the model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Define function for making 2D CNN model for timeseries classification
def make_2d_cnn_model(X_train_shape):
    """Takes the shape of X_train, and outputs a 2d CNN model for greyscale image classification

    Args:
        X_train_shape (tuple): Tuple with dimensions of the training data

    Returns:
        keras.models.Model: 1D CNN model for time series predictions
    """  
    input_shape = (X_train_shape[1], X_train_shape[2], 1) # Get input dimensions to be 28 x 28 x 1

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


# Define function for calculating mean number of active cells per class 
def mean_cells_active(X,y):

    # Retrieve indices for all samples of zeros, ones, two's, etc. ...
    indices = []
    for i in range(10):
        indices.append(list(np.where(y==i)[0]))

    # List of average active cells for each number
    mean_cells_active = []

    # For each number in range 0, 1, 2, .. 9.
    for number in range(len(indices)):
        active_cells_in_images = []

        # For each list of image indices
        for index in indices[number]:
            active_cells_in_img = len(X[index][X[index] > 0])
            active_cells_in_images.append(active_cells_in_img)
        
        mean_cells_active.append(np.mean(active_cells_in_images))
    
    return (mean_cells_active)


# Define function for getting df with counts of active pixels for each image
def active_cells(X,y):
    
    active_cells_in_img = [sum(sum(img> 0)) for img in X]

    df = pd.DataFrame({'class':y, 'change': active_cells_in_img})

    return df

def add_noise(img, n_pixels_change = 125):
    """Function for adding salt and pepper noise

    Args:
        img (array): 2D np.array which one wants to add noise to
        n_pixels_change (int, optional): How many pixels should be changed?. Defaults to 125.

    Returns:
        img (array): 2D array after adding noise
    """    
 
    # Getting the dimensions of the image
    row , col = img.shape
     
    # Randomly pick some pixels in the image for coloring them white. Pick 50
    for i in range(n_pixels_change//2):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord] = 255
         
    # Randomly pick some pixels in
    # the image for coloring them black
    for i in range(n_pixels_change//2):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img

# Function for normalising
def normalize_2d(matrix):
    """Function for normalising elements in an array
    """
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

# Function for finding wronlgy classified mask
def preds_true_false(y: np.array, predictions: np.array):
    """Function for getting list of which classifications are wrong and true 
    
    Args:
        y (array): 1D np.array containing true labels
        predictions (array): 2D np.array contining prediction probabilites

    Returns:
        mask (array): 1D array containing labels 'True' or 'False' 
        indicating the natue of the prediction
    """
    mask = np.array(np.logical_not(np.equal(y, np.argmax(predictions, axis = 1))))

    return mask


# Function for getting array of indexes of wrongly classified images
def wrong_preds(array_1, array_2):
    idxs = []

    for idx, _ in enumerate(array_1):
        if (array_1[idx] == True) and (array_2[idx] == False):
            idxs.append(idx)

    return np.array(idxs)