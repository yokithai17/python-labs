import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets

import typing


def uniform_helper(size=1) -> np.ndarray:
    return np.random.randint(0, 2, size)


coin = uniform_helper


def uniform(size=1, precision=30) -> np.ndarray:
    shape = size if isinstance(size, tuple) else (size,)
    amount_numbers = np.prod(shape)

    bits = coin(amount_numbers * precision)

    binary_array = bits.reshape((amount_numbers, precision))

    power_of_two = np.power((1 / 2), np.arange(1, precision + 1))
    numbers = np.dot(binary_array, power_of_two)

    return np.reshape(numbers, shape)


def show():
    size = 200
    grid = np.linspace(-0.25, 1.25, 500)
    sample = uniform(size, precision=50)

    # Отрисовка графика
    plt.figure(figsize=(10, 4))

    # отображаем значения случайных величин полупрозрачными точками
    plt.scatter(
        sample,
        np.zeros(size),
        alpha=0.4,
        label="Случайная величина"
    )

    # по точкам строим нормированную полупрозрачную гистограмму
    plt.hist(
        sample,
        bins=10,
        density=True,
        alpha=0.4,
        color="orange"
    )

    # рисуем график плотности
    plt.plot(
        grid,
        sps.uniform.pdf(grid),
        color='red',
        linewidth=3,
        label="Плотность случайной величины"
    )

    plt.legend()
    plt.grid(ls=':')
    plt.show()


def analyze():
    size = 100

    plt.figure(figsize=(15, 3))

    for i, precision in enumerate([1, 2, 3, 5, 10, 30]):
        plt.subplot(3, 2, i + 1)
        plt.scatter(
            uniform(size, precision),
            np.zeros(size),
            alpha=0.4
        )
        plt.yticks([])
        if i < 4: plt.xticks([])

    plt.show()