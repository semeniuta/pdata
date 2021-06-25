import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def mvnormal_distrib_map(mu, cov, observations, padding_factor=0.2, grid_prec=0.25):
    """
    Create a discritized map of a multivariate normal distribution on domain
    defined by the observations

    :param mu: Mean of a multivariate normal ((m x 1) array)
    :param cov: Covariance matrix of a multivariate normal ((n x n array))
    :param observations: An (m x n) NumPy array or Pandas data frame,
                         where n is the dimension of the data
                         and m is the number of observations
    :param padding_factor: a factor by which the data span domain is extended
    :param grid_prec: precision of discretization (distance between two consecutive
                      grid points)
    :return:
    """

    dim = observations.shape[1]

    obs_min = np.min(observations, axis=0)
    obs_max = np.max(observations, axis=0)

    dataspan = obs_max - obs_min
    padding = dataspan * padding_factor
    domain = np.array([obs_min - padding, obs_max + padding])

    aranges = [np.arange(domain[0, j], domain[1, j], grid_prec) for j in range(dim)]
    grid_x, grid_y = np.meshgrid(*aranges)

    def make_mv_normal(mu, cov):
        def fn(grid_x, grid_y):

            distrib = np.zeros(grid_x.shape)

            for i in range(grid_x.shape[0]):
                for j in range(grid_x.shape[1]):
                    x = np.array([grid_x[i, j], grid_y[i, j]])
                    distrib[i, j] = multivariate_normal.pdf(x, mu, cov)

            return distrib

        return fn

    mv_normal = make_mv_normal(mu, cov)

    distrib_map = mv_normal(grid_x, grid_y)

    return grid_x, grid_y, distrib_map


def get_axis_extent(grid_x, grid_y):
    """
    Get a tuple of (left, right, bottom, top)
    from the meshgrid output. Used for the extent
    parameter in matplotlib
    """

    left = grid_x[0, 0]
    right = grid_x[0, -1]

    bottom = grid_y[-1, 0]
    top = grid_y[0, 0]

    return left, right, bottom, top


def plot_frozen_distrib(x0, x1, distrib, linspace_num=200, fill_alpha=0.4, **plot_kvargs):
    """
    Visualize 1D continuous distribution on the given domain range
    """

    x = np.linspace(x0, x1, num=linspace_num)
    y = distrib.pdf(x)
    plt.plot(x, y, **plot_kvargs)

    fb_kvargs = {'alpha': fill_alpha}
    if 'color' in plot_kvargs:
        fb_kvargs['color'] = plot_kvargs['color']
    plt.fill_between(x, y, **fb_kvargs)


def handle_fig_and_ax(fig, subplot_pos, projection=None):

    if fig is None:
        fig = plt.figure()

    if subplot_pos is None:
        ax = plt.axes(projection=projection)
    else:
        ax = fig.add_subplot(subplot_pos, projection=projection)

    return fig, ax


def plot_surface(x, y, z, fig=None, subplot_pos=None, **surface_kwargs):

    fig, ax = handle_fig_and_ax(fig, subplot_pos, projection='3d')

    ax.plot_surface(x, y, z, **surface_kwargs)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_contour(x, y, z, fig=None, subplot_pos=None, **contourf_kwargs):

    fig, ax = handle_fig_and_ax(fig, subplot_pos)

    ax.set_aspect('equal')
    cnt = ax.contourf(x, y, z, **contourf_kwargs)
    fig.colorbar(cnt, ax=ax)


def plot_grad_as_vector_field(x_range, y_range, U, V, fig=None, subplot_pos=None, **quiver_kwargs):

    fig, ax = handle_fig_and_ax(fig, subplot_pos)

    ax.set_aspect('equal')
    q = ax.quiver(x_range, y_range, U, V, **quiver_kwargs)


def compact_plotter(n, figsize, n_cols=2):

    for start in range(0, n, n_cols):

        plt.figure(figsize=figsize)
        
        for i in range(n_cols):

            if start + i >= n:
                break

            ax = plt.subplot(1, n_cols, i+1)

            yield ax


class CompactPlotter:

    def __init__(self, plot_func, figsize, n_cols=2):
        self.plot_func = plot_func
        self.figsize = figsize
        self.n_cols = n_cols

    def plot(self, elements, **plot_func_kwargs):

        n = len(elements)

        for start in range(0, n, self.n_cols):

            plt.figure(figsize=self.figsize)
            
            for i in range(self.n_cols):

                if start + i >= n:
                    break

                idx = start + i

                ax = plt.subplot(1, self.n_cols, i+1)

                element = elements[idx]
                self.plot_func(element, **plot_func_kwargs)
