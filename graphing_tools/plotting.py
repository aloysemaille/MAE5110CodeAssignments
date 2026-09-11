import numpy as np
import matplotlib.pyplot as plt

def plot_xy(x,y_series,labels=None,xlabel="",ylabel="",title="",
           diagonal=False,marker_point=None,hlines=None,ax=None):
    if ax is None:
        _,ax=plt.subplots(figsize=(6,5))

    labels=labels or [None]*len(y_series)
    for y,label in zip(y_series,labels):
        ax.plot(x,y,"o-",markersize=3,label=label)

    if diagonal:
        ax.plot(x,x,"--",label="identity line")

    if marker_point is not None:
        ax.plot(*marker_point,"*",markersize=15,label="fixed point")

    for h in (hlines or []):
        ax.axhline(h,linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if any(labels) or diagonal or marker_point is not None:
        ax.legend()