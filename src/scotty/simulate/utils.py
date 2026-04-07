import numpy as np
import numpy.typing as npt


# def double_well_2d_potential(X: npt.NDArray):
#     x = X[..., 0]
#     y = X[..., 1]
#     return 0.5 * (x**2 - 9)**2 + 2 * (y + 1)**2
#

# def double_well_2d_drift(X: npt.NDArray):
#     x = X[..., 0]
#     y = X[..., 1]
#     u = 2 * x * (x ** 2 - 9)
#     v = 4 * (y + 1)
#     return np.stack((u, v), axis=-1)
#

def double_well_2d_potential(X: npt.NDArray):
    x = X[..., 0]
    y = X[..., 1]
    return 0.5 * (x**2 - 9 * (1 - 0.2 * y))**2 + 4 * (y + 2)**2


def double_well_2d_drift(X: npt.NDArray):
    x = X[..., 0]
    y = X[..., 1]
    u = 2 * x * (x ** 2 - 9 * (1 - 0.2 * y))
    v = 8 * (y + 2) + 2 * (x ** 2 - 9 * (1 - 0.2 * y)) * 1.8
    return np.stack((u, v), axis=-1)


def double_well_2d_init(size):
    rng = np.random.default_rng(585)
    return rng.multivariate_normal(
        mean=np.array([1, 8]),
        cov=0.5 * np.eye(2),
        size=size,
    )
