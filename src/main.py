import cv2, keras, os
import numpy as np
import pandas as pd
import seaborn as sb
import csv
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn import metrics, datasets, preprocessing

from utils import (load_data, Q, context, make_1d_cnn_model, make_2d_cnn_model, make_3d_cnn_model, mean_cells_active, active_cells, add_noise)
from rules import cumulative_change
from models.models import (cnn, lr, ensemble_predictions, ensemble_avg_proba)
from plotting.plot_utils import (plot_avg_timeseries, plot_end_dist, plot_mnist, plot_8_generations)

# Defining main function
def main():

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset = "Fashion MNIST", subset=100)
    
    # Generate noisy data
    # X_train_noise = np.array([add_noise(x) for x in X_train])
    # X_val_noise = np.array([add_noise(x) for x in X_val])
    # X_test_noise = np.array([add_noise(x) for x in X_test])

    # Set parameters
    n_gens = 30
    l=0.1
    v=5
    s=500

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
    
    # Make iterations 3D to arrays
    augmented_numbers_cor = np.array(augmented_numbers_cor)
    augmented_numbers_gol = np.array(augmented_numbers_gol)
    augmented_numbers_melt = np.array(augmented_numbers_melt)
    augmented_numbers_cor_val = np.array(augmented_numbers_cor_val)
    augmented_numbers_gol_val = np.array(augmented_numbers_gol_val)
    augmented_numbers_melt_val = np.array(augmented_numbers_melt_val)

    # Make base models
    model_cnn1 = make_1d_cnn_model(cor_mass.shape)

    model_cnn1_melt = make_1d_cnn_model(melt_mass.shape)

    model_cnn1_gol = make_1d_cnn_model(gol_change.shape)

    model_cnn2 = make_2d_cnn_model(X_train.shape)

    model_cnn3 = make_3d_cnn_model(augmented_numbers_gol.shape)

    # Fit models to data
    cnn1_cor_preds, cnn1_cor_report = cnn(cor_mass, y_train, cor_mass_val, y_val, model_cnn1, 50)

    cnn1_gol_preds, cnn1_gol_report = cnn(gol_change, y_train, gol_change_val, y_val, model_cnn1_gol, 50)

    cnn1_melt_preds, cnn1_melt_report = cnn(melt_mass, y_train, melt_mass_val, y_val, model_cnn1_melt, 50)
    
    cnn2_mnist_preds, cnn2_mnist_report = cnn(X_train, y_train, X_val, y_val, model_cnn2, 50)

    # Fit 3D models to data
    cnn3_gol_preds, cnn3_gol_report = cnn(augmented_numbers_gol, y_train, augmented_numbers_gol_val, y_val, model_cnn3, 50)
    cnn3_cor_preds, cnn3_cor_report = cnn(augmented_numbers_cor, y_train, augmented_numbers_cor_val, y_val, model_cnn3, 50)
    cnn3_melt_preds, cnn3_melt_report = cnn(augmented_numbers_melt, y_train, augmented_numbers_melt_val, y_val, model_cnn3, 50)

    # See performances of 3D models
    classif_gol_3d = pd.DataFrame.from_dict(metrics.classification_report(y_val, np.argmax(cnn3_gol_preds, axis=1), output_dict=True))
    print(classif_gol_3d["macro avg"]["f1-score"])

    classif_cor_3d = pd.DataFrame.from_dict(metrics.classification_report(y_val, np.argmax(cnn3_cor_preds, axis=1), output_dict=True))
    print(classif_cor_3d["macro avg"]["f1-score"])

    classif_melt_3d = pd.DataFrame.from_dict(metrics.classification_report(y_val, np.argmax(cnn3_melt_preds, axis=1), output_dict=True))
    print(classif_melt_3d["macro avg"]["f1-score"])

    # Ensemble predictions
    certain_model_gol, digits_gol = ensemble_predictions([cnn1_gol_preds, cnn2_mnist_preds])
    certain_model_cor, digits_cor = ensemble_predictions([cnn1_cor_preds, cnn2_mnist_preds])
    certain_model_melt, digits_melt = ensemble_predictions([cnn1_melt_preds, cnn2_mnist_preds])
    certain_model_full, digits_full = ensemble_predictions([cnn1_cor_preds, cnn1_gol_preds, cnn1_melt_preds,cnn2_mnist_preds])

    ensemble_preds_gol = ensemble_avg_proba([cnn1_gol_preds, cnn2_mnist_preds])
    ensemble_preds_cor = ensemble_avg_proba([cnn1_cor_preds, cnn2_mnist_preds])
    ensemble_preds_melt = ensemble_avg_proba([cnn1_melt_preds, cnn2_mnist_preds])
    ensemble_preds_full = ensemble_avg_proba([cnn1_gol_preds, cnn1_cor_preds, cnn1_melt_preds, cnn2_mnist_preds])

    # Matthews correlation coefficient
    mcc_mnist_certain = metrics.matthews_corrcoef(y_val, np.argmax(cnn2_mnist_preds, axis=1), sample_weight=None)
    mcc_mnist_gol_certain = metrics.matthews_corrcoef(y_val, digits_gol, sample_weight=None)
    mcc_mnist_cor_certain = metrics.matthews_corrcoef(y_val, digits_cor, sample_weight=None)
    mcc_mnist_melt_certain = metrics.matthews_corrcoef(y_val, digits_melt, sample_weight=None)
    mcc_full_certain = metrics.matthews_corrcoef(y_val, digits_full, sample_weight=None)

    mcc_mnist_avg = metrics.matthews_corrcoef(y_val, np.argmax(cnn2_mnist_preds, axis=1), sample_weight=None)
    mcc_mnist_gol_avg = metrics.matthews_corrcoef(y_val, ensemble_preds_gol, sample_weight=None)
    mcc_mnist_cor_avg = metrics.matthews_corrcoef(y_val, ensemble_preds_cor, sample_weight=None)
    mcc_mnist_melt_avg = metrics.matthews_corrcoef(y_val, ensemble_preds_melt, sample_weight=None)
    mcc_full_avg = metrics.matthews_corrcoef(y_val, ensemble_preds_full, sample_weight=None)

    # Classification reports
    classif_mnist_certain    = pd.DataFrame.from_dict(metrics.classification_report(y_val, np.argmax(cnn2_mnist_preds, axis=1), output_dict=True))
    classif_mnist_certain_gol = pd.DataFrame.from_dict(metrics.classification_report(y_val, digits_gol, output_dict=True))
    classif_mnist_certain_cor = pd.DataFrame.from_dict(metrics.classification_report(y_val, digits_cor, output_dict=True))
    classif_mnist_certain_melt = pd.DataFrame.from_dict(metrics.classification_report(y_val, digits_melt, output_dict=True))
    classif_certain_ensemble = pd.DataFrame.from_dict(metrics.classification_report(y_val, digits_full, output_dict=True))
    
    classif_mnist_avg    = pd.DataFrame.from_dict(metrics.classification_report(y_val, np.argmax(cnn2_mnist_preds, axis=1), output_dict=True))
    classif_mnist_avg_gol = pd.DataFrame.from_dict(metrics.classification_report(y_val, ensemble_preds_gol, output_dict=True))
    classif_mnist_avg_cor = pd.DataFrame.from_dict(metrics.classification_report(y_val, ensemble_preds_cor, output_dict=True))
    classif_mnist_avg_melt = pd.DataFrame.from_dict(metrics.classification_report(y_val, ensemble_preds_melt, output_dict=True))
    classif_avg_ensemble = pd.DataFrame.from_dict(metrics.classification_report(y_val, ensemble_preds_full, output_dict=True))
    
    print(classif_mnist_certain["macro avg"]["f1-score"])
    print(classif_mnist_certain_gol["macro avg"]["f1-score"])
    print(classif_mnist_certain_cor["macro avg"]["f1-score"])
    print(classif_mnist_certain_melt["macro avg"]["f1-score"])
    print(classif_certain_ensemble["macro avg"]["f1-score"])

    print(classif_mnist_avg["macro avg"]["f1-score"])
    print(classif_mnist_avg_gol["macro avg"]["f1-score"])
    print(classif_mnist_avg_cor["macro avg"]["f1-score"])
    print(classif_mnist_avg_melt["macro avg"]["f1-score"])
    print(classif_avg_ensemble["macro avg"]["f1-score"])

    # Which models were used?
    pd.Series(certain_model_gol).value_counts()
    pd.Series(certain_model_cor).value_counts()
    pd.Series(certain_model_melt).value_counts()
    pd.Series(certain_model_full).value_counts()


    # Generate other plots


    plot_end_dist(df=df_cor, n_gens=int(n_gens/2))

    plot_end_dist(df=df_gol, n_gens=int(n_gens/5))

    plot_end_dist(df=df_melt, n_gens=int(n_gens))


    # Plotting class-averaged time series for cumulative corrosion mass
    plot_avg_timeseries(df=df_cor)

    plot_avg_timeseries(df=df_gol)
    
    plot_avg_timeseries(df=df_melt)

    # Get active cells in raw mnist
    active_cells = active_cells(X_train,y_train)
    
    plot_end_dist(df=active_cells)

    # Hyperparameter tuning loop

    # Save files
    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/cor_aug.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_cor)
    
    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/gol_aug.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_gol)

    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/melt_aug.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_melt)

    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/cor_aug_val.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_cor_val)
    
    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/gol_aug_val.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_gol_val)

    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/melt_aug_val.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_melt_val)

if __name__ == "__main__":
    main()