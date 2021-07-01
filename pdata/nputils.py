import numpy as np


def index_of_min(x):
    return np.unravel_index(np.argmin(x), x.shape)


def index_of_max(x):
    return np.unravel_index(np.argmax(x), x.shape)
