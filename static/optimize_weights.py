import time
import warnings
import numpy as np
from scipy.integrate import quad
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit

from functions import Polynomial, VelOscillator, GPRealization
from static.quadrature import Quadrature

"""
This file contains some routines to find the optimal weights for quadrature using three given nodes [0, 0.5, 1].
"""


def test_sinusoid():
    # Train a Linear Regression model for integrating Sinusoid functions.
    # The inputs are the function values at the 3 nodes [0, .5, 1] and the output should be the integral.
    # The model should be compared to the Simpsons rule: weights=[1/6, 4/6, 1/6]

    # Create training data of the form x = [f(0), f(.5), f(1)], y = [I(f)]
    set_size = 100000
    x = []
    y = []
    for n in range(set_size):
        f = VelOscillator()
        x.append([f(0), f(.5), f(1)])
        y.append([f.integral()])

    # build Linear Regression model
    reg = LinearRegression().fit(x, y)
    print(reg.coef_)
    print(reg.intercept_)

    # test against simpsons rule
    num_tests = 10000
    reg_better = 0
    total_err_simps = 0.0
    total_err_reg = 0.0
    quadratic_error_simps = 0
    quadratic_error_reg = 0
    for i in range(num_tests):
        f = VelOscillator()
        int_reg = reg.predict([[f(0), f(.5), f(1)]])
        int_reg = int_reg[0][0]
        int_simpsons = 1 / 6 * f(0) + 4 / 6 * f(.5) + 1 / 6 * f(1)
        int_true = f.integral()
        total_err_simps += abs(int_simpsons - int_true)
        total_err_reg += abs(int_reg - int_true)
        quadratic_error_simps += (int_simpsons - int_true) ** 2
        quadratic_error_reg += (int_reg - int_true) ** 2
        if abs(int_reg - int_true) < abs(int_simpsons - int_true):
            reg_better += 1

    print("The Regression Model was better in {} of {} cases".format(reg_better, num_tests))
    print("The average error of the Regression model was {}".format(total_err_reg / num_tests))
    print("The average error of the simpsons rule was {}".format(total_err_simps / num_tests))
    print("The quadratic error of the Regression model was {}".format((quadratic_error_reg / num_tests) ** 0.5))
    print("The quadratic error of the simpsons rule was {}".format((quadratic_error_simps / num_tests) ** 0.5))


def test_polynomial(degree=5):
    # Train a Linear Regression model for integrating Polynomial functions.
    # The inputs are the function values at the 3 nodes [0, .5, 1] and the output should be the integral.
    # The model should be compared to the Simpsons rule: weights=[1/6, 4/6, 1/6]

    # Create training data of the form x = [f(0), f(.5), f(1)], y = [I(f)]
    set_size = 1000000
    x = []
    y = []
    for n in range(set_size):
        f = Polynomial(degree)
        x.append([f(0), f(.5), f(1)])
        y.append([f.integral()])

    # build Linear Regression model
    start = time.time()
    reg = LinearRegression().fit(x, y)
    end = time.time()
    print(end - start)
    print(reg.coef_)
    print(reg.intercept_)

    # test against simpsons rule
    num_tests = 100000
    reg_better = 0
    total_err_simps = 0.0
    total_err_reg = 0.0
    quadratic_error_simps = 0
    quadratic_error_reg = 0
    for i in range(num_tests):
        f = Polynomial(degree)
        int_reg = reg.predict([[f(0), f(.5), f(1)]])
        int_reg = int_reg[0][0]
        int_simpsons = 1 / 6 * f(0) + 4 / 6 * f(.5) + 1 / 6 * f(1)
        int_true = f.integral()
        total_err_simps += abs(int_simpsons - int_true)
        total_err_reg += abs(int_reg - int_true)
        quadratic_error_simps += (int_simpsons - int_true) ** 2
        quadratic_error_reg += (int_reg - int_true) ** 2
        if abs(int_reg - int_true) < abs(int_simpsons - int_true):
            reg_better += 1

    print("The Regression Model was better in {} of {} cases".format(reg_better, num_tests))
    print("The average error of the Regression model was {}".format(total_err_reg / num_tests))
    print("The average error of the simpsons rule was {}".format(total_err_simps / num_tests))
    print("The quadratic error of the Regression model was {}".format((quadratic_error_reg / num_tests) ** 0.5))
    print("The quadratic error of the simpsons rule was {}".format((quadratic_error_simps / num_tests) ** 0.5))


def test_GPRealization():
    # Train a Linear Regression model for integrating functions created by gaussian processes.
    # The inputs are the function values at the 3 nodes [0, .5, 1] and the output should be the integral.
    # The model should be compared to the Simpsons rule: weights=[1/6, 4/6, 1/6]

    # Create training data of the form x = [f(0), f(.5), f(1)], y = [I(f)]
    # kernel = RBF(length_scale=0.5)
    kernel = Matern()
    set_size = 1000
    x = []
    y = []
    for n in range(set_size):
        if n % 10 == 0:
            print(n)
        f = GPRealization(kernel)
        x.append([f(0), f(.5), f(1)])
        integral = quad(f, 0, 1)[0]
        y.append([integral])

    # build Linear Regression model
    reg = LinearRegression().fit(x, y)
    print(reg.coef_)
    print(reg.intercept_)

    # test against simpsons rule
    num_tests = 100
    reg_better = 0
    total_err_simps = 0.0
    total_err_reg = 0.0
    quadratic_error_simps = 0
    quadratic_error_reg = 0
    for i in range(num_tests):
        f = GPRealization(kernel)
        int_reg = reg.predict([[f(0), f(.5), f(1)]])
        int_reg = int_reg[0][0]
        int_simpsons = 1 / 6 * f(0) + 4 / 6 * f(.5) + 1 / 6 * f(1)
        int_true = quad(f, 0, 1)[0]
        total_err_simps += abs(int_simpsons - int_true)
        total_err_reg += abs(int_reg - int_true)
        quadratic_error_simps += (int_simpsons - int_true) ** 2
        quadratic_error_reg += (int_reg - int_true) ** 2
        if abs(int_reg - int_true) < abs(int_simpsons - int_true):
            reg_better += 1

    print("The Regression Model was better in {} of {} cases".format(reg_better, num_tests))
    print("The average error of the Regression model was {}".format(total_err_reg / num_tests))
    print("The average error of the simpsons rule was {}".format(total_err_simps / num_tests))
    print("The quadratic error of the Regression model was {}".format((quadratic_error_reg / num_tests) ** 0.5))
    print("The quadratic error of the simpsons rule was {}".format((quadratic_error_simps / num_tests) ** 0.5))


def expectation():
    """
    Does the same as the routines above, but uses the Quadrature solver instead of scikits LinearRegression to
    obtain the optimal weights.
    Quadrature calculates the Gram matrix A and solves the equation A * lambda = b to obtain the optimum weights.
    This is generally identical to the approach of LinearRegression as both rely on bestapproximation in Hilbert spaces.
    But LinearRegression uses SVD to obtain the solution lambda, so the results may differ slightly.
    Using Quadrature, a lot more info is available, like l2 and mean error, and the covariance and expect. matrices.
    """
    # create data set
    set_size = 10000
    x = np.empty((3, set_size))
    y = np.empty(set_size)
    for n in range(set_size):
        f = Polynomial(5)
        x[:, n] = [f(0.0), f(0.5), f(1.0)]
        y[n] = f.integral()

    start = time.time()
    quadrature = Quadrature(x, y)
    end = time.time()
    print(end - start)
    print("A = {}".format(quadrature.a))
    print("b = {}".format(quadrature.b))
    print("weights = {}".format(quadrature.weights))
    print(quadrature.mean_error)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # test_sinusoid()
        # test_polynomial(degree=6)
        test_GPRealization()
        # expectation()