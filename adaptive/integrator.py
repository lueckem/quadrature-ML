from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from functions import FunctionODE


class Integrator:
    def __call__(self, state, fa):
        """
        Base Integrator call.

        Parameters
        ----------
        state : np.ndarray
            contains step_size and shifted function evaluations f(x) - f(a)
        fa : float
            f(a) to reconstruct the shift

        Returns
        -------
        float
            integral
        """
        return 0


class Simpson(Integrator):
    def __call__(self, state, fa):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and shifted function evaluations f(x) - f(a)
        fa : float
            f(a) to reconstruct the shift

        Returns
        -------
        float
            integral approximated using the simpson rule
        """

        if state.shape[0] != 3:
            raise ValueError('Number of arguments in state has to be 3.')

        step_size = state[0]
        fm, fb = state[1:] + fa
        return 2 * step_size / 6 * (fa + 4 * fm + fb)


class Boole(Integrator):
    def __call__(self, state, fa):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and shifted function evaluations f(x) - f(a)
        fa : float
            f(a) to reconstruct the shift

        Returns
        -------
        float
            integral approximated using boole's rule
        """

        if state.shape[0] != 5:
            raise ValueError('Number of arguments in state has to be 5.')

        step_size = state[0]
        f2, f3, f4, f5 = state[1:] + fa
        return 4 * step_size / 90 * (7 * fa + 32 * f2 + 12 * f3 + 32 * f4 + 7 * f5)


class IntegratorLinReg(Integrator):
    def __init__(self, step_sizes, models):
        """
        Integrator that uses a linear regression model for each step size.

        Parameters
        ----------
        step_sizes : list[float]
        models : list[LinearRegression]
            input are the (unshifted) function evals [f_1,...,f_n] and output the (unshifted integrals)
        """
        self.step_sizes = np.array(step_sizes)
        self.models = models

    def __call__(self, state, fa):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and shifted function evaluations f(x) - f(a)
        fa : float
            f(a) to reconstruct the shift

        Returns
        -------
        float
            approximated integral
        """
        step_size = state[0]
        idx = np.argwhere(np.isclose(self.step_sizes, step_size))[0, 0]
        f_evals = np.concatenate(([fa], state[1:] + fa))
        integral = self.models[idx].predict([f_evals])[0]
        return integral


class StateODE:
    """ small container to handle the states in the ODE case """

    def __init__(self, step_size, f_evals, step_idx=None):
        """
        Parameters
        ----------
        step_size : float
        f_evals : list[np.ndarray]
        step_idx : int, optional
            index of step_size
        """
        self.f_evals = f_evals
        self.step_size = step_size
        self.step_idx = step_idx

    def flatten(self, use_idx=False):
        if not use_idx:
            return np.concatenate(([self.step_size], np.concatenate(self.f_evals)))
        return np.concatenate(([self.step_idx], np.concatenate(self.f_evals)))


class IntegratorODE:
    def __call__(self, state, x0):
        """
        Base Integrator call.

        Parameters
        ----------
        state : StateODE
            contains step_size and function evaluations f(t, x) for fitting inputs t and x
        x0 : np.ndarray
            initial state

        Returns
        -------
        np.ndarray
            state x after step_size
        """
        return 0

    @staticmethod
    def calc_state(t, x, h, f):
        """
        Parameters
        ----------
        t : float
            time
        x : np.ndarray
            state
        h : float
            step size
        f : FunctionODE

        Returns
        -------
        StateODE
            state
        """
        return StateODE(h, [f(x, t)])


class ClassicRungeKutta(IntegratorODE):
    def __call__(self, state, x0):
        """
        Classic Runge Kutta integration

        Parameters
        ----------
        state : StateODE
            contains step_size and function evaluations f(t, x) for fitting inputs t and x
        x0 : np.ndarray
            initial state

        Returns
        -------
        np.ndarray
            state after step_size
        """
        if len(state.f_evals) != 4:
            raise ValueError('Number of f_Evals in state has to be 4.')

        step_size = state.step_size
        k1, k2, k3, k4 = state.f_evals
        return x0 + step_size * (1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4)

    @staticmethod
    def calc_state(t, x, h, f):
        """
        Parameters
        ----------
        t : float
            time
        x : np.ndarray
            state
        h : float
            step size
        f : FunctionODE

        Returns
        -------
        StateODE
            state [h, k1, k2, k3, k4]
        """
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        return StateODE(h, [k1, k2, k3, k4])


class RKDP(IntegratorODE):
    """
    Dormand-Prince RK method of order 5 (6 stages).
    """
    def __init__(self):
        self.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
        self.b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0]
        ])

    def calc_state(self, t, x, h, f):
        """
        Parameters
        ----------
        t : float
            time
        x : np.ndarray
            state
        h : float
            step size
        f : FunctionODE

        Returns
        -------
        StateODE
            state [h, k1, k2, k3, k4]
        """
        k = []
        for i in range(len(self.c)):
            time = t + self.c[i] * h
            state = np.zeros(x.shape)
            for j in range(i):
                state += h * self.a[i, j] * k[j]
            k.append(f(time, x + state))

        return StateODE(h, k)

    def __call__(self, state, x0):
        """
        Dormand-Prince RK method (GoTo solver in ode45)

        Parameters
        ----------
        state : StateODE
            contains step_size and function evaluations f(t, x) for fitting inputs t and x
        x0 : np.ndarray
            initial state

        Returns
        -------
        np.ndarray
            state after step_size
        """
        if len(state.f_evals) != 6:
            raise ValueError('Number of f_Evals in state has to be 6.')

        step_size = state.step_size
        k = state.f_evals
        out = np.zeros(x0.shape)
        for idx in range(len(k)):
            out += step_size * self.b[idx] * k[idx]
        return x0 + out
