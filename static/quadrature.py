import numpy as np


class Quadrature:
    def __init__(self, f_values, integrals):
        """ f_values is a data set of function values at the nodes where one column represents one function.
                integrals is a data set of the corresponding integral values of the form [I_1, ... ].
            calculates all the needed matrices and vectors and finally the optimal weights
        """
        self.num_nodes = np.size(f_values, 0)
        self.covariance_mat = np.cov(f_values, bias=True)
        self.expect_mat = np.mean(f_values, axis=1)
        self.expect_mat = self.expect_mat * np.transpose(self.expect_mat[np.newaxis])
        self.a = self.covariance_mat + self.expect_mat
        self.b = np.matmul(f_values, integrals[:, np.newaxis]) / np.size(f_values, 1)

        self.weights = np.empty((1, self.num_nodes))
        self.weights[0, :] = np.linalg.solve(self.a, self.b[:, 0])

        self.l2_error = np.mean((integrals - np.matmul(self.weights, f_values))**2) ** 0.5

        self.mean_error = np.mean(np.absolute(integrals - np.matmul(self.weights, f_values)))
        self.confidence = self.mean_error / 0.05  # 95 % confidence for error

        self.var_integral = np.var(integrals)

    def predict(self, f_values):
        """ returns vector of predictions for array of f_values shaped like described above. """
        return np.matmul(self.weights, f_values)
