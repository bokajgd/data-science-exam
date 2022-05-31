import cv2, keras, os
import numpy as np
import pandas as pd
import seaborn as sb
import csv
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn import metrics, datasets, preprocessing

from utils import (load_data, Q, context, make_1d_cnn_model, make_1d_cnn_model, make_2d_cnn_model, make_3d_cnn_model, mean_cells_active, active_cells, add_noise, normalize_2d, preds_true_false, wrong_preds, binarise)
from rules import cumulative_change
from models.models import (cnn, lr, ensemble_predictions, ensemble_avg_proba)
from plotting.plot_utils import (plot_avg_timeseries, plot_end_dist, plot_mnist, plot_8_generations, plot_images)

# Defining main function
def main():

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset = "Fashion MNIST", subset=1000)
    X_train_large, X_val_large, X_test_large, y_train_large, y_val_large, y_test_large = load_data(dataset = "Fashion MNIST", subset=20000)

    # Generate noisy data
    # X_train_noise = np.array([add_noise(x) for x in X_train])
    # X_val_noise = np.array([add_noise(x) for x in X_val])
    # X_test_noise = np.array([add_noise(x) for x in X_test])


     ### AUGMENTATION ###   


    # Set parameters
    n_gens = 60
    l=0.1
    v=5
    s=500

    # Generate training features
    df_cor, augmented_numbers_cor, cor_mass = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='corrosion',
                                              n_generations=int(n_gens/2),
                                              Q=Q, 
                                              l=l,
                                              v=v,
                                              context=context)

    df_gol, augmented_numbers_gol, gol_change = cumulative_change(X=X_train, 
                                              y=y_train, 
                                              rule='gol',
                                              n_generations=int(n_gens/5),
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
                                              n_generations=int(n_gens/2),
                                              Q=Q, 
                                              l=l,
                                              v=v,
                                              context=context)

    df_gol_val, augmented_numbers_gol_val, gol_change_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='gol',
                                              n_generations=int(n_gens/5),
                                              threshold=0.0,
                                              context=context)

    df_melt_val, augmented_numbers_melt_val, melt_mass_val = cumulative_change(X=X_val, 
                                              y=y_val, 
                                              rule='melt',
                                              n_generations=n_gens,
                                              s=s,
                                              context=context)

    # Generate features on test set
    df_cor_test, augmented_numbers_cor_test, cor_mass_test = cumulative_change(X=X_test, 
                                              y=y_test, 
                                              rule='corrosion',
                                              n_generations=int(n_gens/2),
                                              Q=Q, 
                                              l=l,
                                              v=v,
                                              context=context)

    df_gol_test, augmented_numbers_gol_test, gol_change_test = cumulative_change(X=X_test, 
                                              y=y_test, 
                                              rule='gol',
                                              n_generations=int(n_gens/5),
                                              threshold=0.0,
                                              context=context)

    df_melt_test, augmented_numbers_melt_test, melt_mass_test = cumulative_change(X=X_test, 
                                              y=y_test, 
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
    augmented_numbers_cor_test = np.array(augmented_numbers_cor_test)
    augmented_numbers_gol_test = np.array(augmented_numbers_gol_test)
    augmented_numbers_melt_test = np.array(augmented_numbers_melt_test)

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

    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/cor_aug_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_cor_test)
    
    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/gol_aug_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_gol_test)

    with open('/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Science 8th Semester/Data Science/Eksamen/data-science-exam/output/melt_aug_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(augmented_numbers_melt_test)

    
    ### RUN MODELS ###

    # Concatenating using augmented images
    X_train_flat = X_train.reshape(2000,784)
    X_test_flat = X_test.reshape(1000,784)

    lr_mnist_preds, lr_mnist_report = lr(X_train, y_train, X_test, y_test)
    lr_mnist_large_preds, lr_mnist_large_report = lr(X_train_large, y_train_large, X_test, y_test)

    last_aug_cor = np.array(augmented_numbers_cor)[:, 30, :,: ]
    last_aug_cor_flat = last_aug_cor.reshape(2000,784)
    last_aug_cor_test = np.array(augmented_numbers_cor_test)[:, 30, :,: ]
    last_aug_cor_test_flat = last_aug_cor_test.reshape(1000,784)

    X_train_aug_cor_con = np.hstack((X_train_flat, last_aug_cor_flat))
    X_test_aug_cor_con = np.hstack((X_test_flat, last_aug_cor_test_flat))

    lr_cor_aug_preds, lr_cor_aug_report = lr(last_aug_cor_flat, y_train, last_aug_cor_test_flat, y_test)
    lr_cor_aug_con_preds, lr_cor_aug_con_report = lr(X_train_aug_cor_con, y_train, X_test_aug_cor_con, y_test)

    last_aug_melt = np.array(augmented_numbers_melt)[:, 60, :,: ]
    last_aug_melt_flat = last_aug_melt.reshape(2000,784)
    last_aug_melt_test = np.array(augmented_numbers_melt_test)[:, 60, :,: ]
    last_aug_melt_test_flat = last_aug_melt_test.reshape(1000,784)

    X_train_aug_melt_con = np.hstack((X_train_flat, last_aug_melt_flat))
    X_test_aug_melt_con = np.hstack((X_test_flat, last_aug_melt_test_flat))

    lr_melt_aug_preds, lr_melt_aug_report = lr(last_aug_melt_flat, y_train, last_aug_melt_test_flat, y_test)
    lr_melt_aug_con_preds, lr_melt_aug_con_report = lr(X_train_aug_melt_con, y_train, X_test_aug_melt_con, y_test)

    last_aug_gol = np.array(augmented_numbers_gol)[:, 12, :,: ]
    last_aug_gol_flat = last_aug_gol.reshape(2000,784)
    last_aug_gol_test = np.array(augmented_numbers_gol_test)[:, 12, :,: ]
    last_aug_gol_test_flat = last_aug_gol_test.reshape(1000,784)

    X_train_aug_gol_con = np.hstack((X_train_flat, last_aug_gol_flat))
    X_test_aug_gol_con = np.hstack((X_test_flat, last_aug_gol_test_flat))

    lr_gol_aug_preds, lr_gol_aug_report = lr(last_aug_gol_flat, y_train, last_aug_gol_test_flat, y_test)
    lr_gol_aug_con_preds, lr_gol_aug_con_report = lr(X_train_aug_gol_con, y_train, X_test_aug_gol_con, y_test)

    X_train_aug_all_con = np.hstack((X_train_flat, last_aug_melt_flat, last_aug_cor_flat, last_aug_gol_flat))
    X_test_aug_all_con = np.hstack((X_test_flat, last_aug_melt_test_flat, last_aug_cor_test_flat, last_aug_gol_test_flat))

    lr_all_aug_con_preds, lr_all_aug_con_report = lr(X_train_aug_all_con, y_train, X_test_aug_all_con, y_test)

    # Plotting MCC
    mcc_lr_mnist = metrics.matthews_corrcoef(y_test, np.argmax(lr_mnist_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_large = metrics.matthews_corrcoef(y_test_large, np.argmax(lr_mnist_large_preds, axis=1), sample_weight=None)
    mcc_lr_melt = metrics.matthews_corrcoef(y_test, np.argmax(lr_melt_aug_preds, axis=1), sample_weight=None)
    mcc_lr_cor = metrics.matthews_corrcoef(y_test, np.argmax(lr_cor_aug_preds, axis=1), sample_weight=None)
    mcc_lr_gol = metrics.matthews_corrcoef(y_test, np.argmax(lr_gol_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_melt = metrics.matthews_corrcoef(y_test, np.argmax(lr_melt_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_cor = metrics.matthews_corrcoef(y_test, np.argmax(lr_cor_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_mnist_gol = metrics.matthews_corrcoef(y_test, np.argmax(lr_gol_aug_con_preds, axis=1), sample_weight=None)
    mcc_lr_all_con = metrics.matthews_corrcoef(y_test, np.argmax(lr_all_aug_con_preds, axis=1), sample_weight=None)

    print(mcc_lr_mnist)
    print(mcc_lr_mnist_large)
    print(mcc_lr_melt)
    print(mcc_lr_cor)
    print(mcc_lr_gol)
    print(mcc_lr_mnist_melt)
    print(mcc_lr_mnist_cor)
    print(mcc_lr_mnist_gol)
    print(mcc_lr_all_con)


    ### PLOTTING ###


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

    
    # Plot wrongly classified
    lr_all_aug_con_true_false = preds_true_false(y_val, lr_all_aug_con_preds)
    lr_mnist_wrong_true_false = preds_true_false(y_val, lr_mnist_preds)
    
    wrong_predictions = wrong_preds(lr_mnist_wrong_true_false,lr_all_aug_con_true_false)
    #print(wrong_predictions) # Number 23 is incorrectly classified

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


    # Plot Sobel gradients
    img = np.array(X_train[12], dtype=np.float32)

    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    plot_images(img)


    # Plot images of 8 iterations 
    cor_first_8 = np.array(augmented_numbers_cor)[0, 0:9, : , :]
    plot_8_generations(cor_first_8)

    gol_first_8 = augmented_numbers_gol[0, 0:9, : , :]
    plot_8_generations(gol_first_8)

    melt_first_8 = augmented_numbers_melt[0, 0:9, : , :]
    plot_8_generations(melt_first_8)


if __name__ == "__main__":
    main()