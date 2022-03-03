import numpy as np


class Reward:
    def __call__(self, error, step_size):
        """
        Parameters
        ----------
        error : float
        step_size : float

        Returns
        -------
        float
            reward
        """
        return 0

    @staticmethod
    def linear_map(point1, point2):
        """
        Calculate parameters of linear map y = a * x + b through two points (x1, y1), (x2, y2).

        Parameters
        ----------
        point1 : tuple[float]
        point2 : tuple[float]

        Returns
        -------
        function
        """
        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - a * point1[0]

        def f(x):
            return a * x + b
        return f

    @staticmethod
    def log_map(point1, point2):
        """
        Calculate logarithmic map y = a * log(b * x) through two points (x1, y1), (x2, y2).

        Parameters
        ----------
        point1 : tuple[float]
        point2 : tuple[float]

        Returns
        -------
        function
        """
        a = (point1[1] - point2[1]) / np.log(point1[0] / point2[0])
        b = np.exp(point2[1] / a) / point2[0]

        def f(x):
            return a * np.log(b * x)
        return f

    @staticmethod
    def exp_map(tol, m, limit):
        """
        Calculate function f(x) = a * exp(-b * x) - L, such that
        1) f(tol) = 0
        2) f(m * tol) = -1
        3) f(inf) = -limit

        Parameters
        ----------
        tol : float
        m : float
        limit : float

        Returns
        -------
        function
        """
        L = limit
        a = (L ** m / (L - 1)) ** (1 / (m - 1))
        b = -np.log(L / a) / tol

        def f(x):
            return a * np.exp(-b * x) - L
        return f


class RewardLog10(Reward):
    def __init__(self, error_tol, step_size_range, reward_range):
        """
        Scale positive rewards logarithmically with step_size.
        Scale negative rewards logarithmically with error.

        Parameters
        ----------
        error_tol : float
        step_size_range : tuple[float]
            lower and upper bound of expected step sizes
        reward_range : tuple[float]
            lower and upper bound of desired reward
        """
        self.error_tol = error_tol
        self.pos_f = self.log_map(*list(zip(step_size_range, reward_range)))

    def __call__(self, error, step_size):
        """
        Parameters
        ----------
        error : float
        step_size : float

        Returns
        -------
        float
            reward
        """
        if error < self.error_tol:
            return self.pos_f(step_size)

        # if error = 10^m * tol, then reward = -m
        return np.log10(self.error_tol / error)


class RewardExp(Reward):
    def __init__(self, error_tol, step_size_range, reward_range):
        """
        Scale positive rewards linearly with step_size.
        Scale negative rewards via negative exponential function w.r.t. error.

        Parameters
        ----------
        error_tol : float
        step_size_range : tuple[float]
            lower and upper bound of expected step sizes
        reward_range : tuple[float]
            lower and upper bound of desired reward
        """
        self.error_tol = error_tol
        self.pos_f = self.linear_map(*list(zip(step_size_range, reward_range)))
        self.neg_f = self.exp_map(error_tol, 2, reward_range[1])

    def __call__(self, error, step_size):
        """
        Parameters
        ----------
        error : float
        step_size : float

        Returns
        -------
        float
            reward
        """
        if error < self.error_tol:
            return self.pos_f(step_size)

        return self.neg_f(error)
