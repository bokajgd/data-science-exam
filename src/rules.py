import numpy as np
import cv2
from typing import Callable, Dict, List, Optional, Union
import pandas as pd

from utils import context


###### CORROSION #####


# Define function for corrosion
def corrosion(seed: np.ndarray, n_generations: int, l: float, v: int, Q):
    """Performs corrosion generations over time, as described in paper by Horsmans, Grøhn & Jessen, 2022

    Args:
        seed (np.ndarray): Image seed, to perform corrosion on - Defaults to np.ndarray.
        n_generations (int): Number of generations to perform. Defaults to int.
        l (float): Number describing how much corrosion takes place
        v (int): Number describing the threshold for "smooth surfaces" (i.e. surfaces where corrosion doesn't happen)
        Q (function): Function that calculates impending corrosion speed - also based on y.
    """
    # Empty list for appending generations to (and start with the seed)
    generations = []

    # Append seed to list of generations
    generations.append(seed)

    # Apply 1-layer reflective-padding
    seed = np.pad(seed, mode="reflect", pad_width=1)

    # Define n_rows and n_cols from shape of img
    n_rows, n_cols = seed.shape

    for generation in range(n_generations):

        # Create image for next step, for overwriting
        generation = np.array(np.zeros(shape=(n_rows, n_cols), dtype=np.int32))

        for r in range(n_rows - 2):
            for c in range(n_cols - 2):

                # Get all neighbours
                context = context(seed, r, c)

                # d (Difference) is difference between center and the lowest in the context
                d = int(seed[r + 1, c + 1]) - int(np.min(context))

                # Any cell with difference > v and difference < 255 changes value to previous_value + q(d, l)
                if 255 >= d and d >= v:
                    generation[r + 1, c + 1] = seed[r + 1, c + 1] + Q(d, l)

                # Any cell with difference smaller than v or with difference larger than 255, then the new generation has the same value as large generation
                if d < v or d > 255:
                    generation[r + 1, c + 1] = seed[r + 1, c + 1]

        # Assign newest generation as the new seed
        seed = generation.copy()

        # Append newest generation to list of generations
        generations.append(generation[1:-1, 1:-1])

    # Return generations
    return generations


###### MELT #####


# Define function for corrosion
def melt(seed: np.ndarray, n_generations: int, s: int = 1000):
    """Performs corrosion generations over time, as described in paper by Horsmans, Grøhn & Jessen, 2022

    Args:
        seed (np.ndarray): Image seed, to perform corrosion on - Defaults to np.ndarray.
        n_generations (int): Number of generations to perform. Defaults to int.
        v (float): Number describing how much corrosion takes place
    """
    # Change dtype of seed
    seed = np.array(seed, dtype=np.float32)

    # Empty list for appending generations to (and start with the seed)
    generations = []

    # Append seed to list of generations
    generations.append(seed)

    # Apply 1-layer reflective-padding
    seed = np.pad(seed, mode="reflect", pad_width=1)

    # Define n_rows and n_cols from shape of img
    n_rows, n_cols = seed.shape

    # Calculate y-direction Sobel image gradient
    sobely = cv2.Sobel(seed, cv2.CV_32F, 0, 1, ksize=3)

    # Augment one generation at a times
    for generation in range(n_generations):

        # Create image for next step, for overwriting
        generation = np.array(np.zeros(shape=(n_rows, n_cols), dtype=np.float32))

        for r in range(n_rows - 1):
            for c in range(n_cols):

                # d (Difference) is pixels image gradient in y direction
                d = sobely[r, c]

                # If gradient is positive, then start melting process
                if d < 0:
                    generation[r + 1, c] = seed[r + 1, c] + (-d / v)

                # else, keep same value
                else:
                    generation[r + 1, c] = seed[r + 1, c]

        # Assign newest generation as the new seed
        seed = generation.copy()

        # Append newest generation to list of generations
        generations.append(generation[1:-1, 1:-1])

        # Calulate sobel gradient
        sobely = cv2.Sobel(seed, cv2.CV_32F, 0, 1, ksize=3)

    # Return generations
    return generations


##### CUMULATIVE MASS #####


# Define function for calculating measure of corrosion-increase-from-baseline on an entire feature set
def cumulative_mass(
    X,
    y,
    rule,
    n_generations,
    Q: Callable = None,
    l: float = None,
    v: int = None,
    s: int = None,
):
    """Function for calculating measure of corrosion-increase-from-baseline on an entire feature set

    Args:
        X (np.nd.array): 3D array with dim(samples, 1st_dimension_of_img, 2nd_dimension_of_img)
        y (np.nd.array): 1D array with labels for images
        rule (str): String specifying which rule to run - takes either 'corrosion' or 'melt'
        n_generations (int): Number of generations to perform
        Q (function): Function that calculates impending corrosion speed - also based on y.
        l (float): Number describing how much corrosion takes place
        v (int): Number describing the threshold for "smooth surfaces" (i.e. surfaces where corrosion doesn't happen)
        s (int, optional): Scaling factor for how much material to melt relative to image gradient
    """
    corrosion_increase_by_number = []
    n_class = []
    augmented_numbers = []

    # For 0, 1, 2, ... len(X):
    for i in range(len(list(X))):

        # Define seed, sum of seed and class of seed
        seed = X[i]
        sum_seed = sum(seed.flatten())
        class_of_seed = y[i]

        # Apply corroosion functions
        if rule == "corrosion":
            generations = corrosion(seed, n_generations, l, v, Q)

        elif rule == "melt":
            generations = melt(seed, n_generations, s)

        # Define lists to store corrosion evoluation and class of current image
        corrosion_increases = []
        c_class = []

        # For each generation, calculate ??? (Jakob, definér?)
        for i in generations:
            sum_generation = sum(i.flatten())

            # Append data to lisrts
            corrosion_increases.append((sum_generation - sum_seed))

            c_class.append(class_of_seed)

        # Append data for each generation to global lists
        corrosion_increase_by_number.append(corrosion_increases)

        n_class.append(c_class)

        augmented_numbers.append(generations)

    # Return cumulative corrosion mass, augmented images and class labels for each image
    return (
        corrosion_increase_by_number,
        augmented_numbers,
        n_class,
    )
