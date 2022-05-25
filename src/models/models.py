import keras
import pandas as pd
from sklearn import metrics

# Defining function for 2D CNN for raw MNIST data

# Defining function for 1D CNN for timeseries classification
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

# Defining ensemble output functions
