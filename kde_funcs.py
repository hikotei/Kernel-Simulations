import numpy as np
from typing import Callable, Optional


"""
These are standard univariate kernels where bandwidth h is not yet implemented.

By default ...

    Gauss   h = infty
    EPA     h = 1
    Window  h = 1

Using simple transformations, we can obtain kernels with specific bandwidths:

K_h = 1/h * K((x_i - x) / h)

"""

def kernel_gauss(x: np.array) -> np.array:
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

def kernel_epa(x: np.array) -> np.array:
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def kernel_window(x: np.array) -> np.array:
    return np.where(np.abs(x) <= 1, 1/2, 0)

def kernel_sinc(x: np.array) -> np.array:
    return np.sin(np.pi * x) / (np.pi * x)


"""
Now extending to multivariate kernels. The simplest multivariate extension is the product kernel:

K(x) = K_1(x_1) * ... * K_d(x_d)

where each K_i is a univariate kernel.

"""

def prod_kernel(x: np.array, kernel: Callable = kernel_epa) -> np.array:

    """
    Returns the product of `kernel` evaluated at each entry of `x`.
    """

    # Apply the kernel function element-wise for each dimension
    res = np.prod(kernel(x), axis=1)
    
    return res

def kde(x: np.array, sample: np.array, kernel: Callable, h: Optional[float]=None) -> float:
    
    """
    Kernel density estimator for `sample` using a `kernel` and optional `h` = bandwidth.
    Returns the estimated density f_kde at a given point / array of points  `x`.

    """

    d = sample.ndim
    n = len(sample)

    if h is None:
        # Silverman's rule of thumb
        h = np.std(sample) * n ** (-1 / 5)

    f_kde = np.zeros(len(x))

    if d > 1:
        for idx, eval_pt in enumerate(x):
            f_kde[idx] = (1/(n * h**d)) * prod_kernel((sample - eval_pt) / h, kernel).sum()
        
    else :
        for idx, eval_pt in enumerate(x):
            f_kde[idx] = (1/(n * h)) * kernel((sample - eval_pt) / h).sum()

    return f_kde