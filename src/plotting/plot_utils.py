import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Defining function for plotting latent time series features averaged over each class
def plot_avg_timeseries(df, x: str='generation', y: str='change', hue: str='class', palette: str='tab10', ylab: str='Change'):

    sb.lineplot(x=x, 
             y=y,
             hue=hue, 
             palette=palette,
             data=df,
            legend=False, ci = 95).set(xlabel = "Generation", ylabel = ylab
            )

    plt.legend(fontsize='small', title_fontsize='12',title='Class', labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker', 'Bag', 'Ankle boot'],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Plot for violin plots
def plot_end_dist(df, n_gens: int=None, x: str='class', y: str='change', palette: str='tab10', ylab: str='Change'):

    if 'generation' in df:

        df = df.loc[df['generation'] == n_gens]

    sb.violinplot(
                x=x, 
                y=y, 
                palette=palette,
                data=df,
                scale='count',
                legend=False).set(xlabel = "Class", ylabel = ylab, xticklabels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker', 'Bag', 'Ankle boot']
                )
    plt.xticks(rotation=45)


# Plot for MNIST examples
def plot_mnist(X, y, cmap = "gray", savename = "plotting/mnist_example.png"):
    """Function for plotting examples of the MNIST dataset

    Args:
        X (array): Array with dim: (n_images, 28, 28)
        y (array): Array with dim: (n_images)
        cmap (str, optional): Which colors to plot with. Defaults to "gray".
        savename (_type_, optional): Output path/name. Defaults to os.path.join("plotting", "mnist_example.png").
    """    
    # Retrieve indices for 10 samples of zeros, ones, two's, etc. ...
    indices = []
    for i in range(10):
        indices.append(list(np.where(y==i)[0][0:10]))

    # Flatten list of indices
    indices = [item for sublist in indices for item in sublist]

    # Plot images with indices and save
    _, axs = plt.subplots(10, 10, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(X[indices], axs):
        ax.imshow(img, cmap=cmap) # Plot image
        ax.axes.xaxis.set_ticklabels([]) # Remove ticklabels for both axes
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([]) # Remove ticks (minor and major) from both axes
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.patch.set_edgecolor('black') # Border around each subplot
        ax.patch.set_linewidth('1')
    plt.savefig(savename)
    plt.show()


# Plot for digit evolution
def plot_8_generations(generations, cmap = "gray", save_name = "generation_plot.png"):
    """Function for plotting the seed and the next 8 generations

    Args:
        generations (array): Array with dimensions:  (n_generations, 28, 28)
        cmap (str, optional): In which colors to plot. Defaults to "gray".
        save_name (_type_, optional): Outputh path for the generated file. Defaults to os.path.join("generation_plot.png").
    """    
    # Plot generations and save
    _, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(generations, axs):
        ax.imshow(img, cmap=cmap)
        ax.axes.xaxis.set_ticklabels([]) # Remove ticklabels for both axes
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([]) # Remove ticks (minor and major) from both axes
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.patch.set_edgecolor('black') # Border around each subplot
        ax.patch.set_linewidth('1')
    plt.savefig(save_name, transparent = True)
    plt.show()


# Plot for wrongly classified
def plot_images(img):
    """Function for plotting an image

    Args:
        img (array): Array with dimensions:  (28, 28)
    """    
    # Plot generations and save
    plt.plot,plt.imshow(img,cmap = 'gray'),plt.xticks([]), plt.yticks([])
