# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:34:08 2020

@author: ryuhei
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_points(x):
    # Normalize x0 and x1 independently to [0, 1]
    x0min = x[:, 0].min()
    x0max = x[:, 0].max()
    x0 = (x[:, 0] - x0min) / (x0max - x0min)
    x1min = x[:, 1].min()
    x1max = x[:, 1].max()
    x1 = (x[:, 1] - x1min) / (x1max - x1min)

    # Each point is colored with (R, G, B) = (x1, 0.0, x0)
    colors = np.stack((x1, np.full(len(x), 0.3), x0), 1)

    fig, ax = plt.subplots(1, 1)

    ax.scatter(*x.T, c=colors)
    ax.set_title('X')
    ax.grid()
    ax.set_aspect('equal')

    plt.show()


def f(x, y):
    denomi = x ** 2 + (1 + y) ** 2
    u = (1 - x ** 2 - y ** 2) / denomi
    v = (2 * x) / denomi
    return u, v


if __name__ == '__main__':
    x = np.linspace(-1, 10, 100)
    y = np.logspace(-5, 10, 100)

    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()
    xy = np.vstack((x, np.log(1 + y)))
    plot_points(xy.T)

    uv = np.vstack(f(x, y))
    plot_points(uv.T)
