import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence


def print_images(
    images: Sequence[np.ndarray],
    titles: Optional[str] = None,
    fig_titles: Optional[Sequence[str]] = None,
) -> None:
    def print_image(image: np.ndarray, fig_title: Optional[str] = None):
        fig, ax = plt.subplots(figsize=plt.figaspect(image))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(image.astype(np.uint8))
        ax.axis("off")
        if fig_title:
            ax.set_title(fig_title)
        plt.show()

    if titles is None:
        titles = [None] * len(images)

    if fig_titles is None:
        fig_titles = [None] * len(images)

    for image, title, fig_title in zip(images, titles, fig_titles):
        print("********************************************")
        print("Data type: ", image.dtype)
        print("Data shape: ", image.shape)
        print("Data value range: ", np.min(image), np.max(image))
        if title:
            print(title)
        print_image(image, fig_title)


if __name__ == "__main__":
    from sklearn.datasets import load_sample_image

    image_names = ["china.jpg", "flower.jpg"]
    images = np.array([load_sample_image(name) for name in image_names])
    print_images(images)
