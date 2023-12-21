import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets

import typing

from part_1 import uniform as coin


def expon(size=1, lambd=1, precision=30):
    sample = coin(size=size, precision=precision)
    lambd = -1 / lambd
    return lambd * np.log(1-sample)


def show():
    size = 100
    grid = np.linspace(-0.5, 5.0, 100)
    sample = expon(size=size, lambd=1)

    # Отрисовка графика
    plt.figure(figsize = (10, 4))

    # отображаем значения случайных величин полупрозрачными точками
    plt.scatter(
        sample,
        np.zeros(size),
        alpha = 0.4,
        label = "Случайная величина"
    )

    # по точкам строим нормированную полупрозрачную гистограмму
    plt.hist(
        sample,
        bins = 10,
        density = True,
        alpha = 0.4,
        color = "orange"
    )

    # рисуем график плотности
    plt.plot(
        grid,
        sps.expon.pdf(grid,scale=1),
        color = 'red',
        linewidth = 3,
        label = "Плотность случайной величины"
    )

    plt.legend()
    plt.grid(ls=':')
    plt.show()
