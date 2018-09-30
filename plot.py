import numpy as np


def heat(f, ax, selection, xaxis, yaxis, zaxis,
         title=None, xlabel=None, ylabel=None,
         xscale='linear', yscale='linear'):
    x = selection[xaxis].values
    y = selection[yaxis].values
    z = selection[zaxis].values

    xs, ys = np.unique(x), np.unique(y)
    X, Y = np.meshgrid(xs, ys)
    Z = z.reshape(xs.shape[0], ys.shape[0]).transpose()

    img = ax.pcolormesh(X, Y, Z)
    f.colorbar(img, ax=ax)

    ax.set_xlabel(xlabel if xlabel else xaxis)
    ax.set_ylabel(ylabel if ylabel else yaxis)
    ax.set_title(title if title else zaxis)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    return ax


