from typing import Optional, Sequence
import matplotlib.pyplot as plt


def print_images_in_subfigures(
    images: Sequence[Sequence[int]],
    labels: Optional[Sequence[Optional[int]]],
    subtitle_prefix: str = "",
    title: str = "",
    figsize: Sequence[float] = (15, 8),
) -> None:
    n_images = len(images)
    n_cols = 5
    n_rows = round(n_images / n_cols)
    if labels is None:
        labels = [None] * n_images
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    for ax, img, label in zip(axs.ravel(), images, labels):
        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"{subtitle_prefix + ': ' if subtitle_prefix else ''}{label}", fontsize="x-large", fontweight="bold"
        )

    for ax in axs.ravel():
        ax.axis("off")

    fig.suptitle(title, fontsize="xx-large", fontweight="bold")
