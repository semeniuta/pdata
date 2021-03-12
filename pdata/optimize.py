import numpy as np


def eval_grad_on_meshgrid(grad_func, grid_x, grid_y):

    xy = np.array([grid_x, grid_y])

    _, nx, ny = xy.shape

    vals = np.zeros_like(xy)

    for i in range(nx):
        for j in range(ny):
            xy_vec = xy[:, i, j]
            g = grad_func(xy_vec)
            vals[:, i, j] = g

    return vals