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

def kernel_epa_hole(x: np.array) -> np.array:
    # epa kernel but with a hole at the origin
    # ie the kernel is zero at the origin
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0) - np.where(x == 0, 3/4, 0)

def kernel_rect(x: np.array) -> np.array:
    return np.where(np.abs(x) <= 1, 1/2, 0)

def kernel_tri(x: np.array) -> np.array:
    return np.where(np.abs(x) <= 1, 1-np.abs(x), 0)

def kernel_sinc(x: np.array) -> np.array:
    return np.where(x == 0, (1/np.pi), np.sin(x)/(np.pi*x))

    # np.where still returns runtime warning for division by zero ...
    # np.where just selects between two arrays after those were computed, so after the warning already occurred
    
    # Q = should i divide x by pi in the argument of sin
    # Q = what to do when d >= 2 ??? divide by pi^d ??? where and when

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

def nadaraya_watson(x_obs, y_obs, x_pred, kernel, h=1):

    """
    VL - Nichtparametrische Stat
    Def 4.2 Nadaraya-Watson-Schätzer

    f_{n, h}^{NW} (x) = \frac {\sum Y_i K(x-x_i)}{\sum K(x-x_i)}}

    x_obs = observation points
    y_obs = observation values
    x_pred = evaluation points
    kernel = kernel function
    h = bandwidth
    """

    # if h is a scalar, make it an array of size x_pred
    if np.isscalar(h):
        h = np.full(x_pred.shape[0], h)

    y_pred = np.zeros(x_pred.shape[0])

    if kernel == "epa":
        kernel_func = kernel_epa
    if kernel == "gauss":
        kernel_func = kernel_gauss
    if kernel == "rect":
        kernel_func = kernel_rect
    if kernel == "tri":
        kernel_func = kernel_tri
    if kernel == "sinc":
        kernel_func = kernel_sinc

    # for each evaluation point

    # w = evaluate kernel function of each observation at evaluation point
    # w * y = multiply with respective y_obs
    # y_pred = w * y / sum(w)

    for i, x in enumerate(x_pred):
        
        w = kernel_func((x - x_obs) / h[i])
        # sum_of_w = np.sum(w)
        # if np.sum(w) is 0 then set sum_of_w to 1

        sum_of_w = np.sum(w) if (sum_of_w := np.sum(w)) != 0 else np.finfo(float).eps

        y_pred[i] = np.sum(w * y_obs / sum_of_w)

    return y_pred

def loc_polynomial_estim_wls(x_obs, y_obs, x_pred, kernel, order, h=1):
    """
    Local polynomial estimator
    """

    # if h is a scalar, make it an array of size x_pred
    if np.isscalar(h):
        h = np.full(x_pred.shape[0], h)

    y_pred = np.empty(x_pred.shape[0])
    y_pred[:] = np.nan

    if kernel == "epa":
        kernel_func = kernel_epa
    elif kernel == "gauss":
        kernel_func = kernel_gauss
    elif kernel == "rect":
        kernel_func = kernel_rect
    elif kernel == "tri":
        kernel_func = kernel_tri
    elif kernel == "sinc":
        kernel_func = kernel_sinc
    else:
        raise ValueError("Unknown kernel type")

    n = x_obs.shape[0]
    num_coefficients = order + 1

    # save the coefficients
    res_coefficients = np.zeros((x_pred.shape[0], num_coefficients))

    for i, x in enumerate(x_pred):

        # create X matrix
        X = np.ones((x_obs.shape[0], num_coefficients))
        for j in range(1, num_coefficients):
            X[:, j] = ((x_obs - x) / h[i]) ** j

        # create Kernel matrix
        K = np.diag(kernel_func((x_obs - x) / h[i]))

        # create Y matrix
        Y = y_obs.reshape(n, 1)

        mat1_temp = np.array(X.T @ K @ X)
        # pseudo inverse using SVD ... more stable for small matrices
        mat1 = np.linalg.pinv(mat1_temp)

        matmul = mat1 @ X.T @ K @ Y
        coefficients = matmul.flatten()

        res_coefficients[i, :] = coefficients
    
    y_pred = res_coefficients[:,0]
    rest = res_coefficients[:,1:]

    return y_pred, rest
