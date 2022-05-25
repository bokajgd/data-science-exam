import keras
import pandas as pd
from sklearn import metrics

# Defining function for CNN models
def cnn(X_train, y_train, X_test, y_test, model, epochs):
    """Trains and tests a given 1D CNN model

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
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if len(X_test.shape) > 2:
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
