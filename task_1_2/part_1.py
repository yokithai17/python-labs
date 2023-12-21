import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
import time

import typing


def matrix_multiplication(A, B):
    return np.sum(A[:, None, :] * np.transpose(B)[None, :, :], axis=-1)


def work_check():
    A = sps.uniform.rvs(size=(10, 20))
    B = sps.uniform.rvs(size=(20, 30))
    print(np.abs(matrix_multiplication(A, B) - A @ B).sum())
