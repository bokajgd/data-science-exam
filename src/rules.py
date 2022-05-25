import pandas as pd
import numpy as np
import cv2
from typing import Callable, Dict, List, Optional, Union
import pandas as pd

from utils import context, binarise


###### GoL #####


# Define function for Game of Life
def GoL(seed=np.ndarray, n_generations=int, context: Callable = context):
    """Performs Game of Life simulations

    Args:
        seed (np.ndarray): Image seed, to perform GoL on  Defaults to np.ndarray.
        n_generations (int): Number of generations to perform. Defaults to int.
        context (function): Function specifying the neighbourhood kernel for a given cell

    Returns:
        List: List of generations
    """
    # Binarise seed
    seed = binarise(seed)

    # Empty list for appending generations to (and start with the seed)
    generations = []

    # Append seed to list of generations
    generations.append(seed)

    # Apply 1-layer 0-padding
    seed = np.pad(seed, 1)

    # Define n_rows and n_cols from shape of img
    n_rows, n_cols = seed.shape

    # Perform ticks
    for i in range(n_generations):
        # Create image for next step, for overwriting
        generation = np.array(np.zeros(shape=(n_rows, n_cols), dtype=np.int32))

        # For loop that iterates over each cell in the array
        for r in range(n_rows - 2):
            for c in range(n_cols - 2):

                # Find number of alive neighbours for each cell
                sum_context = sum(context(seed, r, c).flatten())

                # Any live cell with fewer than 2 or more than 3, dies
                if seed[r + 1, c + 1] == 1 * 255:
                    if sum_context < 2 * 255 or sum_context > 3 * 255:
                        generation[r + 1, c + 1] = 0

                # Any live cell with two or three live neighbours lives, unchanged
                if seed[r + 1, c + 1] == 1 * 255 and 4 * 255 > sum_context > 1 * 255:
                    generation[r + 1, c + 1] = 1 * 255

                # Any dead cell with exactly 3 three live neighbours will come to life
                if seed[r + 1, c + 1] == 0 and sum_context == 3 * 255:
                    generation[r + 1, c + 1] = 1 * 255

        # Assign newest generation as the new seed
        seed = generation.copy()

        # Append newest generation to list of generations
        generations.append(generation[1:-1, 1:-1])

    return generations


###### CORROSION #####


# Define function for corrosion
def corrosion(seed: np.ndarray, n_generations: int, l: float, v: int, Q, context: Callable = context):
    """Performs corrosion generations over time, as described in paper by Horsmans, Grøhn & Jessen, 2022

    Args:
        seed (np.ndarray): Image seed, to perform corrosion on - Defaults to np.ndarray.
        n_generations (int): Number of generations to perform. Defaults to int.
        l (float): Number describing how much corrosion takes place
        v (int): Number describing the threshold for "smooth surfaces" (i.e. surfaces where corrosion doesn't happen)
        Q (function): Function that calculates impending corrosion speed - also based on y.
        context (function): Function specifying the neighbourhood kernel for a given cell
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
def melt(seed: np.ndarray, n_generations: int, s: int = 1000, context: Callable = context):
    """Performs corrosion generations over time, as described in paper by Horsmans, Grøhn & Jessen, 2022

    Args:
        seed (np.ndarray): Image seed, to perform corrosion on - Defaults to np.ndarray.
        n_generations (int): Number of generations to perform. Defaults to int.
        v (float): Number describing how much corrosion takes place
        context (function): Function specifying the neighbourhood kernel for a given cell
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
                    generation[r + 1, c] = seed[r + 1, c] + (-d / s)

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


##### CUMULATIVE CHANGE #####


# Define function for calculating measure of corrosion-increase-from-baseline on an entire feature set
def cumulative_change(
    X,
    y,
    rule,
    n_generations,
    Q: Callable = None,
    l: float = None,
    v: int = None,
    s: int = None,
    context: Callable = context,
):
    """Function for calculating measure of corrosion-increase-from-baseline on an entire feature set

    Args:
        X (np.nd.array): 3D array with dim(samples, 1st_dimension_of_img, 2nd_dimension_of_img)
        y (np.nd.array): 1D array with labels for images
        rule (str): String specifying which rule to run - takes either 'corrosion' or 'melt' or 'gol'
        n_generations (int): Number of generations to perform
        Q (function): Function that calculates impending corrosion speed - also based on y.
        l (float): Number describing how much corrosion takes place
        v (int): Number describing the threshold for "smooth surfaces" (i.e. surfaces where corrosion doesn't happen)
        s (int, optional): Scaling factor for how much material to melt relative to image gradient
        context (function): Function specifying the neighbourhood kernel for a given cell

    """
    log_change = []
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
            generations = corrosion(seed, n_generations, l, v, Q, context)

        elif rule == "melt":
            generations = melt(seed, n_generations, s, context)

        elif rule == "gol":
            generations = GoL(seed, n_generations, context)

        # Define lists to store corrosion evoluation and class of current image
        c_log = []
        c_class = []

        # For each generation, calculate ??? (Jakob, definér?)
        for i in generations:

            # Calc amount of grayness in current gen
            sum_generation = sum(i.flatten())

            # Append data to lisrts
            c_log.append((sum_generation - sum_seed))

            c_class.append(class_of_seed)

        # Append data for each generation to global lists
        log_change.append(c_log)

        n_class.append(c_class)

        augmented_numbers.append(generations)

        # Generate df
        df1 = (
            pd.DataFrame(np.array(n_class).transpose())
            .melt()
            .drop("variable", axis=1)
            .rename({"value": "class"}, axis=1)
        )
        df2 = (
            pd.DataFrame(np.array(log_change).transpose())
            .melt()
            .drop("variable", axis=1)
            .rename({"value": "change"}, axis=1)
        )

        df = pd.DataFrame([df1["class"], df2["change"]]).transpose()

    # Return cumulative corrosion mass, augmented images and class labels for each image
    return (
        df,
        augmented_numbers,
    )
