import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt

# Defining function for plotting latent time series features averaged over each class
def plot_avg_timeseries(df, x: str='generation', y: str='change', hue: str='class', palette: str='tab10'):

    sb.lineplot(x=x, 
             y=y,
             hue=hue, 
             palette=palette,
             data=df)

# Plot for violin plots
def plot_end_dist(df, n_gens, x: str='class', y: str='change', hue: str='class', palette: str='tab10'):

    df = df.loc[df['generation'] == n_gens+1]

    sb.violinplot(
                x="class", 
                y="change", 
                hue=hue, 
                palette=palette,
                data=df)


# Plot for MNIST examples

# Plot for digit evolution

# Plot showing original, augmented and cumulative change

# Plot for hyperparameter tuning

# Plot for wrongly classified

# Plot with image gradient vectors showing in pixels
