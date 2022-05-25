import cv2, keras
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt

from utils import load_data, Q, context, make_1d_cnn_model
from rules import cumulative_change
from plotting.plot_utils import plot_avg_timeseries

# Defining main function
def main():

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Generate features
    df, augmented_numbers = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='corrosion',
                                              n_generations=12,
                                              Q=Q, 
                                              l=0.1,
                                              v=10,
                                              context=context)

    # Generate performance table and plot

    # Generate other plots
    
    # Plotting class-averaged time series for cumulative corrosion mass
    plot_avg_timeseries(df=df)

    # Hyperparametertuning loop


if __name__ == "__main__":
    main()