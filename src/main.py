import cv2, keras, os
import numpy as np
import pandas as pd
import seaborn as sb
import csv
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn import metrics, datasets, preprocessing

from utils import (load_data, Q, context, make_1d_cnn_model, make_1d_cnn_model, make_2d_cnn_model, mean_cells_active, active_cells, add_noise, normalize_2d, preds_true_false, wrong_preds, binarise)
from rules import cumulative_change
from models.models import (cnn, lr, ensemble_predictions, ensemble_avg_proba)
from plotting.plot_utils import (plot_avg_timeseries, plot_end_dist, plot_mnist, plot_8_generations, plot_images)

# Defining main function
def main():

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset = "Fashion MNIST", subset=2000)
    X_train_large, X_val_large, X_test_large, y_train_large, y_val_large, y_test_large = load_data(dataset = "Fashion MNIST", subset=20000)


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


    plot_end_dist(df=df_cor, n_gens=int(n_gens/2), ylab=f'Cumulative Corroded Mass {int(n_gens/2)} Generations')

    plot_end_dist(df=df_gol, n_gens=int(n_gens/5), ylab=f'Change in Living Cells after {int(n_gens/5)} Generations')

    plot_end_dist(df=df_melt, n_gens=int(n_gens), ylab=f'Cumulative Melted Mass {int(n_gens)} Generations')


    # Plotting class-averaged time series for cumulative corrosion mass
    plot_avg_timeseries(df=df_cor, ylab='Cumulative Corroded Mass')

    plot_avg_timeseries(df=df_gol, ylab='Change in Living Cells')
    
    plot_avg_timeseries(df=df_melt, ylab='Cumulative Melted Mass')

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


    # Logistic regression
    # Reshape and flatten
    # Normalising and conctatinating 
    X_train_flat = X_train.reshape(40000,784)
    X_train_flat_norm = normalize_2d(X_train_flat)

    X_val_flat = X_val.reshape(5000,784)
    X_val_flat_norm = normalize_2d(X_val_flat)

    melt_mass_norm = normalize_2d(melt_mass/1000)
    melt_mass_val_norm = normalize_2d(melt_mass_val/1000)

    cor_mass_norm = normalize_2d(cor_mass/1000)
    cor_mass_val_norm = normalize_2d(cor_mass_val/1000)

    gol_change_norm = normalize_2d(gol_change)
    gol_change_val_norm = normalize_2d(gol_change_val)

    X_train_melt_con = np.hstack((X_train_flat, melt_mass/1000))
    X_val_melt_con = np.hstack((X_val_flat, melt_mass_val/1000))

    X_train_con = np.hstack((X_train_flat, melt_mass/1000, cor_mass/1000, gol_change))
    X_val_con = np.hstack((X_val_flat, melt_mass_val/1000, cor_mass_val/1000, gol_change_val))

    lr_mnist_preds, lr_mnist_report = lr(X_train, y_train, X_val, y_val)
    lr_mnist_preds_melt, lr_mnist_report_melt = lr(X_train_melt_con, y_train, X_val_melt_con, y_val)
    lr_mnist_preds_con, lr_mnist_report_con = lr(X_train_con, y_train, X_val_con, y_val)


    # Only last values  
    melt_last = np.array(df_melt['change'].loc[df_melt['generation'] == n_gens])/1000
    melt_last = melt_last.reshape(2000,1)
    cor_last = np.array(df_cor['change'].loc[df_cor['generation'] == n_gens/2])/1000
    cor_last = cor_last.reshape(2000,1)
    gol_last = np.array(df_gol['change'].loc[df_gol['generation'] == n_gens/5])
    gol_last = gol_last.reshape(2000,1)

    melt_last_val = np.array(df_melt_val['change'].loc[df_melt_val['generation'] == n_gens])/1000
    melt_last_val = melt_last_val.reshape(1000,1)
    cor_last_val = np.array(df_cor_val['change'].loc[df_cor_val['generation'] == n_gens/2])/1000
    cor_last_val = cor_last_val.reshape(1000,1)
    gol_last_val = np.array(df_gol_val['change'].loc[df_gol_val['generation'] == n_gens/5])
    gol_last_val = gol_last_val.reshape(1000,1)

    X_train_con_last = np.hstack((X_train_flat, melt_last, cor_last, gol_last))
    X_train_val_con_last = np.hstack((X_val_flat, melt_last_val , cor_last_val, gol_last_val))

    lr_mnist_preds_con_last, lr_mnist_report_con_last = lr(X_train_con_last, y_train, X_train_val_con_last, y_val)

    # On augmentet images
    lr_mnist_preds, lr_mnist_report = lr(X_train, y_train, X_val, y_val)
    lr_mnist_large_preds, lr_mnist_large_report = lr(X_train_large, y_train_large, X_val_large, y_val_large)

    last_aug_cor = np.array(augmented_numbers_cor)[:, 30, :,: ]
    last_aug_cor_flat = last_aug_cor.reshape(4000,784)
    last_aug_cor_val = np.array(augmented_numbers_cor_val)[:, 30, :,: ]
    last_aug_cor_val_flat = last_aug_cor_val.reshape(2000,784)

    X_train_aug_cor_con = np.hstack((X_train_flat, last_aug_cor_flat))
    X_val_aug_cor_con = np.hstack((X_val_flat, last_aug_cor_val_flat))

    lr_cor_aug_preds, lr_cor_aug_report = lr(last_aug_cor_flat, y_train, last_aug_cor_val_flat, y_val)
    lr_cor_aug_con_preds, lr_cor_aug_con_report = lr(X_train_aug_cor_con, y_train, X_val_aug_cor_con, y_val)

    last_aug_melt = np.array(augmented_numbers_melt)[:, 60, :,: ]
    last_aug_melt_flat = last_aug_melt.reshape(4000,784)
    last_aug_melt_val = np.array(augmented_numbers_melt_val)[:, 60, :,: ]
    last_aug_melt_val_flat = last_aug_melt_val.reshape(2000,784)

    X_train_aug_melt_con = np.hstack((X_train_flat, last_aug_melt_flat))
    X_val_aug_melt_con = np.hstack((X_val_flat, last_aug_melt_val_flat))

    lr_melt_aug_preds, lr_melt_aug_report = lr(last_aug_melt_flat, y_train, last_aug_melt_val_flat, y_val)
    lr_melt_aug_con_preds, lr_melt_aug_con_report = lr(X_train_aug_melt_con, y_train, X_val_aug_melt_con, y_val)

    last_aug_gol = np.array(augmented_numbers_gol)[:, 12, :,: ]
    last_aug_gol_flat = last_aug_gol.reshape(4000,784)
    last_aug_gol_val = np.array(augmented_numbers_gol_val)[:, 12, :,: ]
    last_aug_gol_val_flat = last_aug_gol_val.reshape(2000,784)

    X_train_aug_gol_con = np.hstack((X_train_flat, last_aug_gol_flat))
    X_val_aug_gol_con = np.hstack((X_val_flat, last_aug_gol_val_flat))

    lr_gol_aug_preds, lr_gol_aug_report = lr(last_aug_gol_flat, y_train, last_aug_gol_val_flat, y_val)
    lr_gol_aug_con_preds, lr_gol_aug_con_report = lr(X_train_aug_gol_con, y_train, X_val_aug_gol_con, y_val)

    X_train_aug_all_con = np.hstack((X_train_flat, last_aug_melt_flat, last_aug_cor_flat, last_aug_gol_flat))
    X_val_aug_all_con = np.hstack((X_val_flat, last_aug_melt_val_flat, last_aug_cor_val_flat, last_aug_gol_val_flat))

    lr_all_aug_con_preds, lr_all_aug_con_report = lr(X_train_aug_all_con, y_train, X_val_aug_all_con, y_val)

    # Plotting MCC
    mcc_lr_mnist = metrics.matthews_corrcoef(y_val, np.argmax(lr_mnist_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_large = metrics.matthews_corrcoef(y_val_large, np.argmax(lr_mnist_large_preds, axis=1), sample_weight=None)
    mcc_lr_melt = metrics.matthews_corrcoef(y_val, np.argmax(lr_melt_aug_preds, axis=1), sample_weight=None)
    mcc_lr_cor = metrics.matthews_corrcoef(y_val, np.argmax(lr_cor_aug_preds, axis=1), sample_weight=None)
    mcc_lr_gol = metrics.matthews_corrcoef(y_val, np.argmax(lr_gol_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_melt = metrics.matthews_corrcoef(y_val, np.argmax(lr_melt_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_cor = metrics.matthews_corrcoef(y_val, np.argmax(lr_cor_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_gol = metrics.matthews_corrcoef(y_val, np.argmax(lr_gol_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_all_con = metrics.matthews_corrcoef(y_val, np.argmax(lr_all_aug_con_preds, axis=1), sample_weight=None)

    print(mcc_lr_mnist)
    print(mcc_lr_mnist_large)
    print(mcc_lr_melt)
    print(mcc_lr_cor)
    print(mcc_lr_gol)
    print(mcc_lr_mnist_melt)
    print(mcc_lr_mnist_cor)
    print(mcc_lr_mnist_gol)
    print(mcc_lr_all_con)

    # Plot wrongly classified
    lr_all_aug_con_true_false = preds_true_false(y_val, lr_all_aug_con_preds)
    lr_mnist_wrong_true_false = preds_true_false(y_val, lr_mnist_preds)
    
    wrong_predictions = wrong_preds(lr_mnist_wrong_true_false,lr_all_aug_con_true_false)
    # Number 23 is incorrectly classified

    cor_ex = np.array(augmented_numbers_cor_val)[23, 30, : , :]
    plot_images(cor_ex)
    gol_ex = np.array(augmented_numbers_gol_val)[23, 11, : , :]
    plot_images(gol_ex)
    melt_ex = np.array(augmented_numbers_melt_val)[23, 60, : , :]
    plot_images(melt_ex)
    mnist_ex = np.array(X_val)[23, : , :]
    plot_images(mnist_ex)

    # Plot before, after, change plot
    cor_ex_ba = np.array(augmented_numbers_cor_val)[1, 30, : , :]
    gol_ex_ba = np.array(augmented_numbers_gol_val)[1, 11, : , :]
    melt_ex_ba = np.array(augmented_numbers_melt_val)[1, 60, : , :]
    mnist_ex_ba = np.array(X_val)[1, : , :]

    before_after = []

    cor_change = cor_ex_ba - mnist_ex_ba
    before_after.append(mnist_ex_ba)
    before_after.append(cor_ex_ba)
    before_after.append(cor_change)

    gol_change = gol_ex_ba - binarise(mnist_ex_ba)
    before_after.append(binarise(mnist_ex_ba))
    before_after.append(gol_ex_ba)
    before_after.append(gol_change)

    melt_change = melt_ex_ba - mnist_ex_ba
    before_after.append(mnist_ex_ba)
    before_after.append(melt_ex_ba)
    before_after.append(melt_change)

    before_after = np.array(before_after)

    plot_8_generations(before_after)

if __name__ == "__main__":
    main()