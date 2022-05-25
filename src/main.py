import cv2, keras
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt

from utils import (load_data, Q, context, make_1d_cnn_model, make_1d_cnn_model, make_2d_cnn_model)
from rules import cumulative_change
from models.models import (cnn, lr)
from plotting.plot_utils import (plot_avg_timeseries, plot_end_dist)

# Defining main function
def main():

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(subset=100)

    # Generate training features
    df_cor, augmented_numbers_cor, cor_mass = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='corrosion',
                                              n_generations=15,
                                              Q=Q, 
                                              l=0.1,
                                              v=10,
                                              context=context)

    df_gol, augmented_numbers_gol, gol_change = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='gol',
                                              n_generations=50,
                                              threshold=0.0,
                                              context=context)

    df_melt, augmented_numbers_melt, melt_mass = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='melt',
                                              n_generations=15,
                                              s=100,
                                              context=context)

                                    
    # Generate features on validation set
    df_cor_val, augmented_numbers_cor_val, cor_mass_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='corrosion',
                                              n_generations=15,
                                              Q=Q, 
                                              l=0.1,
                                              v=10,
                                              context=context)

    df_gol_val, augmented_numbers_gol_val, gol_change_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='gol',
                                              n_generations=50,
                                              threshold=0.0,
                                              context=context)

    df_melt_val, augmented_numbers_melt_val, melt_mass_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='melt',
                                              n_generations=15,
                                              s=100,
                                              context=context)
                                              
    # Make base models
    model_cnn1 = make_1d_cnn_model(melt_mass.shape)

    model_cnn2 = make_2d_cnn_model(X_train.shape)

    # Fit models to data
    cnn1_cor_preds, cnn1_cor_report = cnn(cor_mass, y_train, cor_mass_val, y_val, model_cnn1, 50)

    cnn1_gol_preds, cnn1_gol_report = cnn(gol_change, y_train, gol_change_val, y_val, model_cnn1, 50)

    cnn1_melt_preds, cnn1_melt_report = cnn(melt_mass, y_train, melt_mass_val, y_val, model_cnn1, 50)
    
    cnn2_mnist_preds, cnn2_mnist_report = cnn(X_train, y_train, X_train, y_train, model_cnn2, 5)

    # Ensemble prediction

    # Generate other plots
    plot_end_dist(df=df_melt, n_gens=15)
    
    # Plotting class-averaged time series for cumulative corrosion mass
    plot_avg_timeseries(df=df_cor)

    plot_avg_timeseries(df=df_gol)
    
    plot_avg_timeseries(df=df_melt)

    # Hyperparametertuning loop


if __name__ == "__main__":
    main()