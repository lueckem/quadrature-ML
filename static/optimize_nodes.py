import time
import warnings
import numpy as np
from collections import namedtuple
from itertools import combinations
from operator import attrgetter
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize
from scipy.integrate import quad
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, Lasso
from sklearn.feature_selection import RFE

from functions import Polynomial, VelOscillator, GPRealization
from static.quadrature import Quadrature


"""
This file contains some ideas for retrieving a choice of optimal nodes given a set of possible nodes
(-> feature selection)
This problem is really difficult and none of the proposed algorithms performs well.
"""


def test_OMP():
    """
    find the 3 best nodes in the set [0, 0.1, ..., 0.9, 1.0] and their weights using Orthogonal Matching Pursuit
    """
    kernel = Matern(length_scale=0.8, nu=1.2)
    set_size = 100
    x = []
    y = []
    for n in range(set_size):
        f = GPRealization(kernel)
        data = []
        for num in np.linspace(0, 1, 11):
            data.append(f(num))
        x.append(data)
        y.append(quad(f, 0, 1)[0])

    # build OMP model
    reg = OrthogonalMatchingPursuit(3).fit(x, y)
    print(reg.coef_)
    print(reg.intercept_)

    # test against simpsons rule
    num_tests = 100
    reg_better = 0
    total_err_simps = 0.0
    total_err_reg = 0.0
    for i in range(num_tests):
        f = GPRealization(kernel)
        data = []
        for num in np.linspace(0, 1, 11):
            data.append(f(num))
        int_reg = reg.predict([data])
        int_reg = int_reg[0]
        int_simpsons = 1 / 6 * f(0) + 4 / 6 * f(.5) + 1 / 6 * f(1)
        int_true = quad(f, 0, 1)[0]
        total_err_simps += abs(int_simpsons - int_true)
        total_err_reg += abs(int_reg - int_true)
        if abs(int_reg - int_true) < abs(int_simpsons - int_true):
            reg_better += 1

    print("The Regression Model was better in {} of {} cases".format(reg_better, num_tests))
    print("The average error of the Regression model was {}".format(total_err_reg / num_tests))
    print("The average error of the simpsons rule was {}".format(total_err_simps / num_tests))


def calc_error(nodes):
    """ calculate the mean error of a choice of nodes for integrating VelOscillator from -1 to 1. """
    set_size = 1000
    test_set_size = 1000
    x0 = 0
    x1 = 1

    # make model
    num_nodes = np.size(nodes)
    x = np.empty((set_size, num_nodes))
    y = np.empty(set_size)
    for n in range(set_size):
        f = VelOscillator()
        x[n, :] = [f(node) for node in nodes]
        y[n] = f.integral(x0, x1)
    r = LinearRegression().fit(x, y[:, np.newaxis])

    # test model
    x_test = np.empty((test_set_size, num_nodes))
    y_test = np.empty(test_set_size)
    for n in range(test_set_size):
        f = VelOscillator()
        x_test[n, :] = [f(node) for node in nodes]
        y_test[n] = f.integral(x0, x1)

    predictions = r.predict(x_test)[:, 0]
    error = np.mean(np.abs(predictions - y_test))
    print(error)
    return error


def optimize_nodes():
    """ try to find the best nodes using nonlinear optimization """
    nodes = np.array([0.2, 0.8])
    result = minimize(calc_error, nodes, method='Powell')
    print(result)


def experiment_relations_1d(degree):
    """ Quadrature with one single node. Find out the best node by trying out many different nodes.
    Plot the performance of every node and 1 - abs(correlation) -> they have the same minimum. """
    num_nodes = 100
    nodes = np.linspace(0, 1, num_nodes)
    set_size = 5000
    kernel = RBF(length_scale=0.5)

    errors = np.empty(num_nodes)
    correlations = np.empty(num_nodes)

    x = np.empty((num_nodes, 1, set_size))  # one row for one node
    y = np.empty(set_size)

    start = time.time()

    for n in range(set_size):
        f = GPRealization(kernel)
        # f = Polynomial(degree)
        x[:, 0, n] = [f(node) for node in nodes]
        y[n] = f.integral(0, 1)

    for idx in range(num_nodes):
        q = Quadrature(x[idx, :, :], y)
        errors[idx] = q.l2_error
        correlations[idx] = 1 - np.abs(np.corrcoef(x[idx, 0, :], y)[0, 1])

    end = time.time()
    print("took {} seconds".format(end - start))

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=250)

    color = 'tab:red'
    ax1.set_xlabel('Position des Knotens')
    ax1.set_ylabel('error', color=color)
    # ax1.set_ylim([0, 1])
    ax1.plot(nodes, errors, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_title('Polynomgrad = {}'.format(degree))
    ax1.set_title('Gauß Prozess mit RBF-kernel')

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('1 - abs(cor)', color=color)
    # ax2.set_ylim([0, 0.4])
    ax2.plot(nodes, correlations, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    # plt.show()
    plt.savefig("pol{}.png".format(degree))


def experiment_relations():
    """ Go through all the different combinations of num_nodes of num_possible_nodes nodes.
    For every combination estimate the error of the optimal Quadrature. Print the 5 best choices of nodes.
    Plot some metrics relating to the correlation of the nodes and the integral to find a connection -> no success. """
    num_nodes = 3
    num_possible_nodes = 11
    set_size = 10000

    combs = combinations(np.linspace(0, 1, num_possible_nodes), num_nodes)
    combs = list(combs)

    Result = namedtuple('Result', 'nodes weights cov_mat expect_mat A b var_integral error')
    results = []

    for idx, nodes in enumerate(combs):
        x = np.empty((num_nodes, set_size))
        y = np.empty(set_size)
        for n in range(set_size):
            f = Polynomial(6)
            x[:, n] = [f(node) for node in nodes]
            y[n] = f.integral()

        q = Quadrature(x, y)
        results.append(Result(nodes, q.weights, q.covariance_mat, q.expect_mat, q.a, q.b, q.var_integral, q.mean_error))

    # print the best results
    results.sort(key=attrgetter('error'))
    for i in range(5):
        print(results[i].nodes, results[i].error)

    # plot some metrics

    # # ### test covariance --> no correlation
    # x = np.empty(len(results))
    # y = np.empty(len(results))
    # for idx, result in enumerate(results):
    #     cov_mat = np.copy(result.cov_mat)
    #     np.fill_diagonal(cov_mat, 0.0)
    #     x[idx] = np.linalg.norm(cov_mat)
    #     y[idx] = result.error
    #
    # plt.plot(x, y, 'x')
    # plt.show()

    x1 = np.empty(len(results))
    x2 = np.empty(len(results))
    x3 = np.empty(len(results))
    x4 = np.empty(len(results))
    y = np.empty(len(results))
    for idx, result in enumerate(results):
        cor_f_i = np.copy(result.b)
        for i in range(num_nodes):
            cor_f_i[i] /= np.sqrt(result.cov_mat[i, i] * result.var_integral)
        x1[idx] = np.sum(np.abs(cor_f_i))
        cor_mat = np.copy(result.cov_mat)
        np.fill_diagonal(cor_mat, 0.0)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                cor_mat[i, j] /= np.sqrt(result.cov_mat[i, i] * result.cov_mat[j, j])
        x2[idx] = np.sum(np.abs(cor_mat)) / 2
        x3[idx] = np.sum(np.diagonal(result.cov_mat))
        x4[idx] = x1[idx] - x2[idx]
        y[idx] = result.error

    plt.figure()
    plt.plot(x1, y, 'x')
    plt.figure()
    plt.plot(x2, y, 'x')
    plt.figure()
    plt.plot(x3, y, 'x')
    plt.figure()
    plt.plot(x4, y, 'x')
    plt.show()


def error_2d(degree):
    """
    For every possible choice of 2 nodes in [-1, 1] plot the performance of the optimal Quadrature.
    """
    num_nodes = 100
    nodes = np.linspace(0, 1, num_nodes)
    set_size = 10000

    errors = np.zeros((num_nodes, num_nodes))

    x = np.empty((num_nodes, set_size))  # one row for one node
    y = np.empty(set_size)

    start = time.time()

    for n in range(set_size):
        f = VelOscillator()
        x[:, n] = [f(node) for node in nodes]
        y[n] = f.integral(0, 1)

    for idx1 in range(num_nodes):
        for idx2 in range(idx1 + 1, num_nodes):
            # print(x[[idx1, idx2], :])
            q = Quadrature(x[[idx1, idx2], :], y)
            errors[idx1, idx2] = q.l2_error
            errors[idx2, idx1] = q.l2_error

    end = time.time()
    print("took {} seconds".format(end - start))

    print(np.min(errors))

    plt.figure(dpi=250)
    plt.rcParams.update({'font.size': 14})
    plot_coord = np.linspace(0, 1, num_nodes+1)
    plt.pcolor(plot_coord, plot_coord, errors, norm=LogNorm())
    plt.colorbar()
    # plt.title('Fehler bei zwei Stützstellen, Polynom vom Grad {}'.format(degree))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()
    plt.savefig("pol{}.png".format(degree))
    plt.show()


def correlation_matrix():
    """ Calculate the correlation matrix of different nodes and the integral. """
    # create data
    num_nodes = 5
    nodes = np.linspace(0, 1, num_nodes)
    set_size = 100000

    x = np.empty((num_nodes, set_size))  # one row for one node
    y = np.empty(set_size)

    for n in range(set_size):
        f = VelOscillator()
        x[:, n] = [f(node) for node in nodes]
        y[n] = f.integral(0, 1)

    # calculate correlation matrix
    data = np.concatenate((x, np.transpose(y[:, np.newaxis])))
    cor = np.corrcoef(data)
    print(cor)


def lasso():
    """ Using the lasso Linear Model find the best choice of num_nodes_choose nodes. """
    # #### create data #######################
    num_nodes = 20
    num_nodes_choose = (5, 7)  # lower bound, upper bound
    nodes = np.linspace(0, 1, num_nodes)
    set_size = 100000

    x = np.empty((num_nodes, set_size))  # one row for one node
    y = np.empty(set_size)

    for n in range(set_size):
        f = VelOscillator()
        x[:, n] = [f(node) for node in nodes]
        y[n] = f.integral(0, 1)

    # #### train Lasso Model, tune the alpha parameter so that the number of chosen nodes fits ###############
    print('-------- Finding the correct alpha and run LASSO --------')
    alpha = 0.1
    reg = Lasso(alpha=alpha)
    while True:
        reg.fit(np.transpose(x), y)
        num = len(np.nonzero(reg.coef_)[0])
        print('alpha = {}, number of chosen nodes = {}'.format(alpha, num))
        if num < num_nodes_choose[0]:
            alpha = alpha / 2
            reg = Lasso(alpha=alpha)
        elif num > num_nodes_choose[1]:
            alpha = 1.5 * alpha
            reg = Lasso(alpha=alpha)
        else:
            break

    node_indices = np.nonzero(reg.coef_)[0]
    num_nodes_chosen = len(node_indices)
    chosen_nodes = nodes[node_indices]
    print('It chose the nodes {} with weights {}'.format(chosen_nodes, reg.coef_[np.nonzero(reg.coef_)[0]]))

    # calculate error
    error = np.mean((reg.predict(np.transpose(x)) - y) ** 2) ** 0.5
    print('The chosen nodes and weights yield an error of {}'.format(error))
    q = Quadrature(x[node_indices, :], y)
    # print(q.weights)  # Lasso has different weights than Quadrature for the same nodes ???
    # print(reg.coef_)

    # #### find the best possible choice by trying out all the combinations ########################
    # ### CAUTION ####################
    print('')
    print('-------- Finding the best nodes via brute force --------')
    combs = combinations(range(num_nodes), num_nodes_chosen)
    combs = list(combs)
    print('Number of combinations for sets of {} nodes: {}'.format(num_nodes_chosen, len(combs)))

    Result = namedtuple('Result', 'nodes weights error')
    results = []

    for idx, node_idx in enumerate(combs):
        # 'node_idx' are the indices of the nodes in this combinations
        q = Quadrature(x[node_idx, :], y)
        results.append(Result(nodes[np.array(node_idx)], q.weights, q.l2_error))

    # print the best results
    results.sort(key=attrgetter('error'))
    for i in range(3):
        print('{}.: {} with error {} and weights {}'.format(i + 1, results[i].nodes, results[i].error, results[i].weights))

    result_of_chosen_nodes = next((r_idx, r) for r_idx, r in enumerate(results) if np.array_equal(r.nodes, chosen_nodes))
    print('The nodes picked by Lasso are the {}.-best nodes with an error of {}'.format(result_of_chosen_nodes[0] + 1,
                                                                                        result_of_chosen_nodes[1].error))

    # #### compare to equidistant nodes #######################################
    print('')
    print('-------- Solve the problem with the same number of equidistant nodes --------')
    nodes_equi = np.linspace(0, 1, num_nodes_chosen)
    x_equi = np.empty((num_nodes_chosen, set_size))  # one row for one node
    y_equi = np.empty(set_size)
    for n in range(set_size):
        f = VelOscillator()
        x_equi[:, n] = [f(node) for node in nodes_equi]
        y_equi[n] = f.integral(0, 1)
    q = Quadrature(x_equi, y_equi)
    print('equidistant nodes {} yield an error of {}'.format(nodes_equi, q.l2_error))

    # # plot example functions and the chosen nodes
    # funs = []
    # for i in range(10):
    #     funs.append(VelOscillator())
    #
    # xplot = np.linspace(0, 1, 101)
    # for f in funs:
    #     yplot = []
    #     for num in xplot:
    #         yplot.append(f(num))
    #     plt.plot(xplot, yplot, '-')
    #
    # for xc in chosen_nodes:
    #     plt.axvline(x=xc, color='r')
    #
    # plt.show()


def recursive_feature_elimination():
    """ perform recursive feature elimination on a Linear Regression model to retrieve an optimal choice of nodes
        from a set of nodes. """
    # #### create data #######################
    num_nodes = 20
    num_nodes_choose = 5
    nodes = np.linspace(0, 1, num_nodes)
    set_size = 100000

    x = np.empty((num_nodes, set_size))  # one row for one node
    y = np.empty(set_size)

    for n in range(set_size):
        f = VelOscillator()
        x[:, n] = [f(node) for node in nodes]
        y[n] = f.integral(0, 1)

    # ### perform recursive feature elimination on a Linear Regression model
    reg = LinearRegression()
    rfe = RFE(estimator=reg, n_features_to_select=num_nodes_choose)
    rfe.fit(np.transpose(x), y)
    print('selected the nodes: {}'.format(nodes[rfe.support_]))

    # ### calculate the error
    error = np.mean((rfe.predict(np.transpose(x)) - y) ** 2) ** 0.5
    print('The chosen nodes and weights yield an error of {}'.format(error))

    # the error of 0.0035 is a lot worse than the optimal error of 0.0001
    # -> not a feasible method


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # test_OMP()
        # experiment_relations()
        # experiment_relations_1d(5)
        # optimize_nodes()
        error_2d(0)
