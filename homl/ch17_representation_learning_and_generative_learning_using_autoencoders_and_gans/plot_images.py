import matplotlib.pyplot as plt
from typing import Optional, Sequence

import numpy as np


def plot_images(
    images_norm: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[int],
    label_text: str = "Label",
    n_cols: int = 10,
    sup_title: Optional[str] = None,
) -> None:
    n_images = len(images_norm)
    n_rows = round(np.ceil(n_images / n_cols))
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        figsize=(20, n_rows * 2 + 1),
        constrained_layout=True,
    )
    for ax, img, label in zip(axs.ravel(), images_norm, labels):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{label_text}: {label}", fontsize="large", fontweight="bold")
    if sup_title is not None:
        fig.suptitle(sup_title, fontsize="x-large", fontweight="bold")
