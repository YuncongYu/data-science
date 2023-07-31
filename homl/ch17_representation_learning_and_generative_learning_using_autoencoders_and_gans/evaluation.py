import json
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

import plot_images


class Evaluation:
    def __init__(
        self,
        model: tf.keras.models.Model,
        x_test: Sequence[Sequence[float]],
        y_test: Sequence[int],
    ):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def reconstruction_test(self, n_plots: int = 20) -> None:
        x_test_rec = self.model.predict(self.x_test)

        with open("fashion_mnist_classes.json", "r") as fp:
            self.classes_ = np.array(json.load(fp))

        plot_images.plot_images(
            images_norm=self.x_test[:n_plots],
            labels=self.classes_[self.y_test[:n_plots]],
            sup_title="Original Test Data",
        )

        plot_images.plot_images(
            images_norm=x_test_rec[:n_plots] * 255,
            labels=self.classes_[self.y_test[:n_plots]],
            sup_title="Reconstructed Test Data",
        )

    def dl_plot(self) -> None:
        x_test_embedding = self.model.layers[0].predict(self.x_test)

        tsne = TSNE()
        x_test_2d = tsne.fit_transform(x_test_embedding)

        fig, ax = plt.subplots()
        ax.scatter(x_test_2d[:, 0], x_test_2d[:, 1], c=self.y_test, s=10, cmap="tab10")
        ax.axis("off")
