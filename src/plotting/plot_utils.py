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
def plot_end_dist(df, n_gens: int=None, x: str='class', y: str='change', palette: str='tab10'):

    if 'generation' in df:

        df = df.loc[df['generation'] == n_gens]

    sb.violinplot(
                x="class", 
                y="change", 
                palette=palette,
                data=df,
                scale='count')


# Plot for MNIST examples

# Plot for digit evolution

# Plot showing original, augmented and cumulative change

# Plot for hyperparameter tuning

# Plot for wrongly classified

# Plot with image gradient vectors showing in pixels
