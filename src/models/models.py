import keras
import pandas as pd
import numpy as np
from sklearn import metrics
from keras import layers
from keras.utils import np_utils
from keras.datasets import mnist
from scipy.signal import medfilt
from sklearn.preprocessing import normalize
from sklearn import svm, metrics, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Defining function for CNN models
def cnn(X_train, y_train, X_test, y_test, model, epochs):
    """Trains and tests a given CNN model

    Args:
        X_train (array): Training set with dim = (n_samples, len_timeseries)
        y_train (array): Training labels with dim = (n_samples)
        X_test (array): Training set with dim = (n_samples, len_timeseries)
        y_test (array): Training labels with dim = (n_samples)
        model (keras.models.Model): 1D CNN model (may be retrieved from make_1d_cnn_model())
        epochs (int): Number of epochs for the training

    Returns:
        tuple: Tuple containing predictions (in probabilities) and a classification report
    """    
    # Make y_train into one-hot-encoding
    y_train = keras.utils.np_utils.to_categorical(y_train, 10)

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # Compile the model

    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, verbose = False)

    # Make predictions - outcome is in probabilites
    predictions = model.predict(X_test)

    # Make classification report
    classification_report = pd.DataFrame.from_dict(metrics.classification_report(y_test, np.argmax(predictions, axis=1), output_dict=True))

    return predictions, classification_report


# Defining logistic regression for raw MNIST data
def lr(X_train, y_train, X_test, y_test):
    """Trains and tests a logistic regression. May take 2D or 1D features

    Args:
        X_train (array): Training set with dim = (n_samples, 1st_dim) or dim = (n_samples, 1st_dim, 2nd_dim)
        y_train (array): Training labels with dim = (n_samples)
        X_test (array): Testing set with dim = (n_samples, 1st_dim) or dim = (n_samples, 1st_dim, 2nd_dim)
        y_test (array): Testing labels with dim = (n_samples)

    Returns:
        tuple: Tuple containing predictions (in probabilities) and a classification report
    """    

    # If the array is 2D, flatten to 1D
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if len(X_test.shape) == 3:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # If the array id 3D, flatten to 1D
    if len(X_train.shape) == 4:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if len(X_test.shape) == 4:
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Define model
    model = LogisticRegression(max_iter=250)

    # Fit model
    model.fit(X_train, y_train)

    # Make predictions (and get outcome in probabilities)
    predictions = model.predict_proba(X_test)

    # Make classification report
    classification_report = pd.DataFrame.from_dict(metrics.classification_report(y_test, np.argmax(predictions, axis=1), output_dict=True))

    return predictions, classification_report


# Defining ensemble output functions
def ensemble_predictions(predictions):
    """Takes a list of 2D arrays with predictions. Finds the most certain of the model predictions for each image, as well as its prediction.

    Args:
        predictions (list): List of 2D arrays with predictions

    Returns:
        list: certain_model_indices is a 1D array. Each element shows the index of the model that was most certain for the given trial.
        list: certain_predictions is a 1D array. Each element shows the prediction of the most that was most certain for the given trial.
    """    
    predictions = np.stack(predictions)

    certain_model_indices = []
    certain_predictions = []
    for i in range(predictions.shape[1]):
        
        max_certainty = np.amax(predictions[:, i, :])
        certain_model_index = np.where(predictions[:, i , :] == max_certainty)[0]
        certain_prediction = np.where(predictions[:, i , :] == max_certainty)[1]
        certain_model_indices.append(certain_model_index[0])
        certain_predictions.append(certain_prediction[0])

    return certain_model_indices, certain_predictions

# Defining function for making ensemble predictions with probability averaging
def ensemble_avg_proba(list_of_predict_probabilities):
    """Ensemble prediction using average probability

    Args:
        list_of_predict_probabilities (list): List of arrays with probability predictions - one for each model

    Returns:
        array: Array with average probability predictions
    """    
    
    acc_proba = list_of_predict_probabilities[0]

    for i in list_of_predict_probabilities[1:]:
        
        acc_proba += i

    avg_proba = acc_proba/len(list_of_predict_probabilities)

    ensemble_predictions = np.argmax(avg_proba, axis = 1)

    return ensemble_predictions
