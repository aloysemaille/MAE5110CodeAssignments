import numpy as np
import matplotlib.pyplot as plt

def plot(x_serie, y_series, labels=None, xlabel="", ylabel="", title="",
         diagonal=False, marker_point=None, hlines=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    y_series = np.asarray(y_series)
    if y_series.ndim == 1:
        y_series = [y_series]

    labels = labels if labels is not None else [None] * len(y_series)
    if isinstance(labels, str):
        labels = [labels]

    for y, label in zip(y_series, labels):
        ax.plot(x_serie, y, "o-", markersize=3, label=label)

    if diagonal:
        ax.plot(x_serie, x_serie, "--", label="identity line")

    if marker_point is not None:
        ax.plot(*marker_point, "*", markersize=15, label="fixed point")

    for h in (hlines or []):
        ax.axhline(h, linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if any(labels) or diagonal or marker_point is not None:
        ax.legend()
