import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets

import typing

from part_1 import uniform as coin

def box_muller_transform(size: int, precision) -> np.ndarray:
    if size % 2 == 0:
        flag = False
        size = size//2
    else:
        flag = True
        size = (size + 1)//2

    uniform_array = coin(size=size, precision=precision)

    cos_array = np.cos(2 * np.pi * uniform_array)
    sin_array = np.sin(2 * np.pi * uniform_array)
    log_array = np.log(uniform_array)
    log_array = np.sqrt(-2 * log_array)

    X_array = cos_array * log_array
    Y_array = sin_array * log_array

    arr =  np.concatenate((X_array, Y_array), axis=None)
    arr = arr[:-1] if flag else arr

    return arr


def normal(size=1, loc=0, scale=1, precision=30) -> np.ndarray:
    amount_numbers = size if isinstance(size, tuple) else np.prod(size)
    standart_distribution = box_muller_transform(size=amount_numbers, precision=precision)
    
    return standart_distribution * scale + loc

def show():
    size = 200
    grid = np.linspace(-3.0, 3.0, 200)
    sample = normal(size=200, loc=0, scale=1)

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
        sps.norm.pdf(grid, loc=0, scale=1),
        color = 'red',
        linewidth = 3,
        label = "Плотность случайной величины"
    )

    plt.legend()
    plt.grid(ls=':')
    plt.show()