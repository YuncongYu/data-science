import json
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import plot_images


def load_data(
    n_plots: int = 10,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    (
        (x_train_valid, y_train_valid),
        (x_test, y_test),
    ) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid, y_train_valid, test_size=0.2, random_state=42
    )

    with open("fashion_mnist_classes.json", "r") as fp:
        classes = np.array(json.load(fp))

    plot_images.plot_images(
        images_norm=x_train[:n_plots],
        labels=classes[y_train[:n_plots]],
        sup_title="Original Data",
    )

    x_train_norm = x_train / 255.0
    x_valid_norm = x_valid / 255.0
    x_test_norm = x_test / 255.0

    return (x_train_norm, y_train), (x_valid_norm, y_valid), (x_test_norm, y_test)
