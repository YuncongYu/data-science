from typing import Sequence

import matplotlib.pyplot as plt


def plot(series: Sequence[float], y_true: Sequence[float], y_pred: Sequence[float]):
    n_steps_known = len(series)
    n_steps_pred = len(y_true)
    n_steps_total = n_steps_known + n_steps_pred
    t_pred = range(n_steps_known, n_steps_total)

    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(series, c="tab:blue", marker="o")
    ax.plot(t_pred, y_true, c="tab:blue", marker="x", label="Actual")
    ax.plot(t_pred, y_pred, c="tab:orange", marker="o", label="Prediction")
    ax.grid()
    ax.set_xlim([0, n_steps_total])
    ax.set_ylim([-1, 1])
    ax.set_xlabel("t")
    ax.set_ylabel("X(t)")
    ax.legend()