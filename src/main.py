import cv2, keras
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt
from sklearn import metrics, datasets, preprocessing

from utils import (load_data, Q, context, make_1d_cnn_model, make_1d_cnn_model, make_2d_cnn_model, mean_cells_active, active_cells, add_noise)
from rules import cumulative_change
from models.models import (cnn, lr, ensemble_predictions)
from plotting.plot_utils import (plot_avg_timeseries, plot_end_dist)

# Defining main function
def main():

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(subset=10000)
    
    # Generate noisy data
    X_train_noise = np.array([add_noise(x) for x in X_train])
    X_val_noise = np.array([add_noise(x) for x in X_val])
    X_test_noise = np.array([add_noise(x) for x in X_test])

    # Set parameters
    n_gens = 12
    l=0.1
    v=10
    s=100

    # Generate training features
    df_cor, augmented_numbers_cor, cor_mass = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='corrosion',
                                              n_generations=n_gens,
                                              Q=Q, 
                                              l=l,
                                              v=v,
                                              context=context)

    df_gol, augmented_numbers_gol, gol_change = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='gol',
                                              n_generations=n_gens,
                                              threshold=0.0,
                                              context=context)

    df_melt, augmented_numbers_melt, melt_mass = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='melt',
                                              n_generations=n_gens,
                                              s=s,
                                              context=context)

                                    
    # Generate features on validation set
    df_cor_val, augmented_numbers_cor_val, cor_mass_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='corrosion',
                                              n_generations=n_gens,
                                              Q=Q, 
                                              l=l,
                                              v=v,
                                              context=context)

    df_gol_val, augmented_numbers_gol_val, gol_change_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='gol',
                                              n_generations=n_gens,
                                              threshold=0.0,
                                              context=context)

    df_melt_val, augmented_numbers_melt_val, melt_mass_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='melt',
                                              n_generations=n_gens,
                                              s=s,
                                              context=context)
                                              
    # Make base models
    model_cnn1 = make_1d_cnn_model(melt_mass.shape)

    model_cnn1_gol = make_1d_cnn_model(gol_change.shape)

    model_cnn2 = make_2d_cnn_model(X_train.shape)

    # Fit models to data
    cnn1_cor_preds, cnn1_cor_report = cnn(cor_mass, y_train, cor_mass_val, y_val, model_cnn1, 50)

    cnn1_gol_preds, cnn1_gol_report = cnn(gol_change, y_train, gol_change_val, y_val, model_cnn1_gol, 50)

    cnn1_melt_preds, cnn1_melt_report = cnn(melt_mass, y_train, melt_mass_val, y_val, model_cnn1, 50)
    
    cnn2_mnist_preds, cnn2_mnist_report = cnn(X_train, y_train, X_val, y_val, model_cnn2, 10)

    # Ensemble predictions
    certain_model, digits = ensemble_predictions([cnn1_gol_preds,cnn1_melt_preds,cnn2_mnist_preds])

    classification_report = pd.DataFrame.from_dict(metrics.classification_report(y_val, digits, output_dict=True))

    pd.Series(certain_model).value_counts()

    # Generate other plots
    plot_end_dist(df=df_cor, n_gens=n_gens)

    plot_end_dist(df=df_gol, n_gens=n_gens)

    plot_end_dist(df=df_melt, n_gens=n_gens)


    # Plotting class-averaged time series for cumulative corrosion mass
    plot_avg_timeseries(df=df_cor)

    plot_avg_timeseries(df=df_gol)
    
    plot_avg_timeseries(df=df_melt)

    # Get active cells in raw mnist
    active_cells = active_cells(X_train,y_train)
    
    plot_end_dist(df=active_cells)

    # Hyperparameter tuning loop

if __name__ == "__main__":
    main()