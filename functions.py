import numpy as np
from math import sin, cos, pi, exp
from scipy.special import erf
from scipy.optimize import root_scalar
from scipy.integrate import quad, solve_ivp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, RBF
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Function:
    def __init__(self):
        self.evals = 0

    def reset(self, reset_params=True):
        self.evals = 0

    def __call__(self, x):
        self.evals += 1
        return 0

    def antiderivative(self, x):
        return 0

    def integral(self, x_0=0.0, x_1=1.0):
        return self.antiderivative(x_1) - self.antiderivative(x_0)

    @staticmethod
    def maximum():
        return 0

    @staticmethod
    def minimum():
        return 0


class Sinus(Function):
    """
    f(x) = A*sin(w*x+phi)
    A = 1, w ~ Unif(0, 1.5 pi), phi ~ Unif(0, 2 pi)
    resonable stepsizes: [0.05, 0.75]
    reasonable tolerance: 0.0005
    """

    def __init__(self):
        super().__init__()
        self.amplitude, self.frequency, self.shift = self.choose_params()

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0
        if reset_params:
            self.amplitude, self.frequency, self.shift = self.choose_params()

    @staticmethod
    def choose_params():
        amplitude = 1.0
        frequency = 1.5 * pi * np.random.sample()
        shift = 2.0 * pi * np.random.sample()
        return amplitude, frequency, shift

    def __call__(self, x):
        self.evals += 1
        return self.amplitude * sin(self.frequency * x + self.shift)

    def antiderivative(self, x):
        """ return the antiderivative at x """
        return -self.amplitude * cos(self.frequency * x + self.shift) / self.frequency

    @staticmethod
    def maximum():
        return 1.0

    @staticmethod
    def minimum():
        return -1.0


class SuperposeSinus(Function):
    """
    f(x) = sum_{i=1}^d c_i * A_i * sin(w_i * x + phi_i)
    c_i ~ Unif(0, 1)
    A_i, w_i, phi_i from Sinus
    """

    def __init__(self, num_funs):
        super().__init__()
        self.d = num_funs  # number of sinus waves to be superposed
        self.sinuses = [Sinus() for _ in range(self.d)]
        self.c = self.choose_params(self.d)  # coefficients

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0
        if reset_params:
            self.c = self.choose_params(self.d)
            for sinus in self.sinuses:
                sinus.reset()

    @staticmethod
    def choose_params(d):
        return np.random.rand(d)

    def __call__(self, x):
        self.evals += 1
        out = 0.0
        for i in range(self.d):
            out += self.c[i] * self.sinuses[i](x)
        return out

    def antiderivative(self, x):
        """ return the antiderivative at x """
        out = 0.0
        for i in range(self.d):
            out += self.c[i] * self.sinuses[i].antiderivative(x)
        return out

    def maximum(self):
        return self.d

    def minimum(self):
        return -self.d


class Sinusoid(Function):
    """ f(x) = A*sin(w*x+phi)*exp(-D*x)
        A ~ Unif(0,1)
        w ~ Unif(0,2pi)
        phi ~ Unif(0,2pi)
        D ~ Unif(0,3)
    """

    def __init__(self):
        super().__init__()
        self.amplitude, self.frequency, self.shift, self.damping = self.choose_params()

    def reset(self, reset_params=True):
        self.evals = 0
        if reset_params:
            self.amplitude, self.frequency, self.shift, self.damping = self.choose_params()

    @staticmethod
    def choose_params():
        amplitude = 1.0  # np.random.sample()
        frequency = 1.5 * pi * np.random.sample()
        shift = 2.0 * pi * np.random.sample()
        damping = 0.0  # 3 * np.random.sample()
        return amplitude, frequency, shift, damping

    def __call__(self, x):
        return self.amplitude * sin(self.frequency * x + self.shift) * exp(-self.damping * x)

    def antiderivative(self, x):
        """ return the antiderivative at x """
        out = -self.amplitude * exp(-self.damping * x)
        out *= self.damping * sin(self.frequency * x + self.shift) + self.frequency * cos(
            self.frequency * x + self.shift)
        out /= self.damping * self.damping + self.frequency * self.frequency
        return out

    @staticmethod
    def maximum():
        return 1.0

    @staticmethod
    def minimum():
        return -1.0


class VelOscillator(Function):
    """ f(t) = velocity of oscillator with amplitude, frequency, shift and damping
        A ~ Unif(0,1)
        w ~ Unif(0,2pi)
        phi ~ Unif(0,2pi)
        D ~ Unif(0,1)
        The function value is set to 1 outside of [low_bound, up_bound]
    """

    def __init__(self, low_bound=-1, up_bound=1):
        super().__init__()

        self.low_bound = low_bound
        self.up_bound = up_bound
        self.amplitude, self.frequency, self.shift, self.damping = self.choose_params()

    def reset(self, reset_params=True):
        self.evals = 0
        if reset_params:
            self.amplitude, self.frequency, self.shift, self.damping = self.choose_params()

    @staticmethod
    def choose_params():
        amplitude = np.random.sample()
        frequency = 2 * pi * np.random.sample()
        shift = 2 * pi * np.random.sample()
        damping = 1 * np.random.sample()
        return amplitude, frequency, shift, damping

    def __call__(self, x):
        if x < self.low_bound or x > self.up_bound:
            return 1

        out = self.damping * sin(self.frequency * x + self.shift) - self.frequency * cos(
            self.frequency * x + self.shift)
        return -self.amplitude * exp(-self.damping * x) * out

    def antiderivative(self, x):
        return self.amplitude * sin(self.frequency * x + self.shift) * exp(-self.damping * x)

    @staticmethod
    def maximum():
        return 1.0

    @staticmethod
    def minimum():
        return -1.0


class Pulse(Function):
    """ f(x) = c * e^(-a * (x - b)^2), a > 0"""

    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = self.choose_params()

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0
        if reset_params:
            self.a, self.b, self.c = self.choose_params()

    @staticmethod
    def choose_params():
        a = (6000 * np.random.sample() - 3000) + 10000
        b = 6 * np.random.sample() + 2
        c = (6000 * np.random.sample() - 3000) + 10000
        return a, b, c

    def __call__(self, x):
        self.evals += 1
        return self.c * exp(-self.a * (x - self.b) ** 2)

    def antiderivative(self, x):
        """ return the antiderivative at x """
        return self.c * -(pi ** 0.5 * erf(self.a ** 0.5 * (self.b - x))) / (2 * self.a ** 0.5)

    @staticmethod
    def maximum():
        return 1.0

    @staticmethod
    def minimum():
        return 0.0


class Polynomial(Function):
    """ f(x) = c_0 + c_1 * x + ... + c_d * x^d
        d = 5
        designed to be evaluated at values in [-1, 1] """

    def __init__(self, degree=5):
        super().__init__()
        self.d = degree
        self.c = self.choose_params()  # coefficients

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0
        if reset_params:
            self.c = self.choose_params()

    def choose_params(self):
        return np.random.rand(self.d + 1) * 2 - 1.0

    def __call__(self, x):
        self.evals += 1
        out = 0.0
        for i in range(self.d + 1):
            out += self.c[i] * x ** i
        return out

    def antiderivative(self, x):
        """ return the antiderivative at x """
        out = 0.0
        for i in range(self.d + 1):
            out += self.c[i] / (i + 1) * x ** (i + 1)
        return out

    def maximum(self):
        return self.d

    def minimum(self):
        return -self.d


class BrokenPolynomial(Function):
    def __init__(self):
        super().__init__()
        self.d = 5  # degree
        self.c = self.choose_params()
        self.condition = 1  # break polynomial if derivative = condition
        self.breakpt = self.find_breakpoint()

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0
        if reset_params:
            self.c = self.choose_params()
            self.breakpt = self.find_breakpoint()

    def choose_params(self):
        return np.random.rand(self.d + 1) * 2 - 1.0

    def derivative(self, x):
        out = 0.0
        for i in range(1, self.d + 1):
            out += self.c[i] * i * x ** (i - 1)
        return out

    def find_breakpoint(self):
        breakpt = 2
        for x0 in np.linspace(-1, 1, 11):
            sol = root_scalar(lambda x: (self.derivative(x) - self.condition), x0=x0, x1=x0 + 0.01)
            if sol.converged:
                if -1 < sol.root < 1:
                    breakpt = sol.root
                    break
        return breakpt

    def __call__(self, x):
        self.evals += 1
        if x > self.breakpt:
            return 0

        out = 0
        for i in range(self.d + 1):
            out += self.c[i] * x ** i
        return out

    def antiderivative(self, x):
        if x > self.breakpt:
            x = self.breakpt

        out = 0.0
        for i in range(self.d + 1):
            out += self.c[i] / (i + 1) * x ** (i + 1)
        return out

    def maximum(self):
        return self.d

    def minimum(self):
        return -self.d


class StaggeredSinus(Function):
    """ Sinus with high frequency, then middle, then low for each one period. """

    def __init__(self):
        super().__init__()
        self.a1 = 1.25
        self.a2 = 0.85
        self.a3 = 0.5

        self.sinus1 = Sinus()
        self.sinus1.shift = 0.0
        self.sinus1.frequency = self.a1 * pi

        self.sinus2 = Sinus()
        self.sinus2.shift = -2 * pi * self.a2 / self.a1
        self.sinus2.frequency = self.a2 * pi

        self.sinus3 = Sinus()
        self.sinus3.shift = -2 * pi * self.a3 * (1.0 / self.a1 + 1.0 / self.a2)
        self.sinus3.frequency = self.a3 * pi

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0

    def __call__(self, x):
        self.evals += 1
        if x < 2 / self.a1:
            return self.sinus1(x)
        if x < 2 / self.a1 + 2 / self.a2:
            return self.sinus2(x)
        return self.sinus3(x)

    def antiderivative(self, x):
        """ return the antiderivative at x """
        if x < 2 / self.a1:
            return self.sinus1.antiderivative(x)
        if x < 2 / self.a1 + 2 / self.a2:
            return self.sinus2.antiderivative(x)
        return self.sinus3.antiderivative(x)

    def integral(self, x_0=0.0, x_1=1.0):
        """ return the integral from x_0 to x_1 """
        switch2 = 2 / self.a1 + 2 / self.a2
        if x_1 < 2 / self.a1:
            return self.sinus1.integral(x_0, x_1)
        if x_0 < 2 / self.a1:
            integ = self.sinus1.integral(x_0, 2 / self.a1)
            if x_1 < switch2:
                return integ + self.sinus2.integral(2 / self.a1, x_1)
            integ += self.sinus2.integral(2 / self.a1, switch2)
            return integ + self.sinus3.integral(switch2, x_1)
        if x_1 < switch2:
            return self.sinus2.integral(x_0, x_1)
        if x_0 < switch2:
            return self.sinus2.integral(x_0, switch2) + self.sinus3.integral(switch2, x_1)
        return self.sinus3.integral(x_0, x_1)

    @staticmethod
    def maximum():
        return 1.0

    @staticmethod
    def minimum():
        return -1.0

    @property
    def start(self):
        return 0.0

    @property
    def end(self):
        return 2 * pi * self.sinus3.frequency / self.sinus2.frequency - self.sinus2.shift - self.sinus1.shift


class GPRealization(Function):
    """ A callable class that is the realization of a gaussian process, f:R->R. """

    def __init__(self, kernel=None, x0=0, x1=1, num_init=100):
        """
        Parameters
        ----------
        kernel : Kernel
        x0 : float
            left boundary
        x1 : float
            right boundary
        num_init : int
            the number of equidistant samples between x0 and x1 used to build the function
        """
        super().__init__()
        # RBF(length_scale_bounds=(0.3, 10.0))
        kernel = Matern(length_scale=1)
        kernel = RBF(length_scale=0.1)
        self.gp = GaussianProcessRegressor(kernel=kernel)

        x = np.linspace(x0, x1, num_init)
        x = x[:, np.newaxis]
        y = self.gp.sample_y(x, n_samples=1, random_state=np.random.RandomState())
        y = y[:, 0]
        self.gp.fit(x, y)

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0
        if reset_params:
            raise NotImplementedError('No reset of params implemented for GPRealization')

    def __call__(self, x):
        return self.gp.predict([[x]])[0]

    def integral(self, x_0=0, x_1=1):
        """ return the integral from x_0 to x_1 """
        return quad(self, x_0, x_1)[0]


class DoublePendulumInteg(Function):
    def __init__(self, x0, x1):
        super().__init__()
        self.x0, self.x1 = x0, x1
        y_init = DoublePendulum().sample_initial_x(energy=20)
        sol = solve_ivp(DoublePendulum(), (x0, x1), y_init, atol=1e-6, rtol=1e-8, dense_output=True)
        self.traj = sol.sol

    def reset(self, reset_params=True):
        """ parameters are chosen """
        self.evals = 0
        if reset_params:
            y_init = DoublePendulum().sample_initial_x(energy=20)
            sol = solve_ivp(DoublePendulum(), (self.x0, self.x1), y_init, atol=1e-6, rtol=1e-8, dense_output=True)
            self.traj = sol.sol

    def __call__(self, x):
        return self.traj(x)[1]

    def integral(self, x_0=0, x_1=1):
        """ return the integral from x_0 to x_1 """
        limit = max(50, int((x_1 - x_0) * 50))
        return quad(self, x_0, x_1, limit=limit, epsrel=1e-10, epsabs=1e-10)[0]


class FunctionODE:
    def __init__(self):
        self.evals = 0

    def reset(self, reset_params=True):
        self.evals = 0

    def __call__(self, t, x):
        self.evals += 1
        return np.zeros(x.shape)

    def solve(self, t_0, x_0, t_1, t_eval=None):
        return x_0


class Rotation(FunctionODE):
    """ A rotation in 2 dimensions, i.e. f(t,x)=[[b a][-a b]] x """

    def __init__(self):
        super().__init__()
        self.A = self.choose_params()

    @staticmethod
    def choose_params():
        a = np.random.random() + 0.5
        a = 1
        b = -0.1
        return np.array([[b, a], [-a, b]])

    def reset(self, reset_params=True):
        self.evals = 0
        if reset_params:
            self.A = self.choose_params()

    def __call__(self, t, x):
        """
        f(t,x)=[[0 a][-a 0]] x

        Parameters
        ----------
        t : float
        x : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self.evals += 1
        return self.A @ x

    def solve(self, t_0, x_0, t_1, t_eval=None):
        """
        return solution x(t_1) of IVP x(t_0) = x_0

        Parameters
        ----------
        t_0 : float
        x_0 : np.ndarray
        t_1 : float
        t_eval : np.ndarray, optional
            intermediate values at which the state x is calculated and returned

        Returns
        -------
        np.ndarray
        """
        a = self.A[0, 1]
        b = self.A[0, 0]

        if t_eval is None:
            c = np.cos(a * (t_1 - t_0))
            s = np.sin(a * (t_1 - t_0))
            return np.exp(b * (t_1 - t_0)) * np.array([[c, s], [-s, c]]) @ x_0

        t_out = np.concatenate(([t_0], t_eval, [t_1]))
        x_out = np.zeros((len(t_out), len(x_0)))
        for idx, t in enumerate(t_out):
            c = np.cos(a * (t - t_0))
            s = np.sin(a * (t - t_0))
            x_out[idx, :] = np.exp(b * (t - t_0)) * np.array([[c, s], [-s, c]]) @ x_0
        return t_out, x_out


# class Pendulum(FunctionODE):
#     """ A damped pendulum, which is bumped if the energy gets low  """
#
#     def __init__(self, switchpoints=(-np.infty, np.infty), delay=0.1):
#         """
#         Parameters
#         ----------
#         switchpoints : Tuple[float], optional
#             if bump and ||f|| > switchpoints[1]: switch to free
#             if free and ||f|| < switchpoints[0]: switch to bump
#         delay : float, optional
#             the time that the system needs to exceed the switchpoint before a switch takes place
#         """
#         super().__init__()
#         self.switchpoints = switchpoints
#         self.delay = delay
#
#         self.switch_times = []
#         self.t_delay = []  # store times of delayed points
#         self.x_delay = []  # store norm of delayed points
#         self.free = True  # always start free
#
#     def reset(self, reset_params=True):
#         self.evals = 0
#         self.switch_times = []
#         self.t_delay = []
#         self.x_delay = []
#         self.free = True
#
#     def __call__(self, t, x):
#         """
#         Parameters
#         ----------
#         t : float
#         x : np.ndarray
#
#         Returns
#         -------
#         np.ndarray
#         """
#         self.evals += 1
#
#         # find out if the system is free or bumped at time t
#         last_idx = len(self.switch_times) - 1
#         idx = last_idx
#         while idx >= 0:
#             if t >= self.switch_times[idx]:
#                 break
#             idx -= 1
#         this_free = self.free if (last_idx - idx) % 2 == 0 else not self.free
#
#         # calculate f(x, t)
#         if this_free:
#             out = np.array([[-0.1, 1], [-1, -0.1]]) @ x
#         else:
#             out = np.array([0, 10 * (t - self.switch_times[idx]) + 3])
#
#         # calculate the average over delayed values
#         abs_out = np.linalg.norm(out)
#         self.t_delay.append(t)
#         self.x_delay.append(abs_out)
#         # self.t_delay = list(dict.fromkeys(self.t_delay))  # make unique
#         # self.x_delay = list(dict.fromkeys(self.x_delay))  # make unique
#         sorted_idx = sorted(range(len(self.t_delay)), key=self.t_delay.__getitem__)
#         # print(self.t_delay)
#         # print(self.x_delay)
#         self.t_delay = [self.t_delay[idx] for idx in sorted_idx]
#         self.x_delay = [self.x_delay[idx] for idx in sorted_idx]
#         idx = 0
#         while self.t_delay[-1] - self.t_delay[idx] > self.delay:
#             idx += 1
#         # if idx == len(self.t_delay) - 1:  # keep 2 elements
#         #     idx -= 1
#         self.t_delay = self.t_delay[idx:]
#         self.x_delay = self.x_delay[idx:]
#         if len(self.t_delay) > 1:
#             avg_delay = np.trapz(self.x_delay, self.t_delay) / (self.t_delay[-1] - self.t_delay[0])
#         else:
#             avg_delay = self.x_delay[0]
#         # print(self.t_delay[0], self.t_delay[-1], len(self.t_delay))
#         # print(len(np.unique(self.t_delay)))
#         print(avg_delay, t)
#         # print('')
#
#         # switching
#         if t >= self.t_delay[-1] and self.t_delay[-1] - self.t_delay[0] > 0.5 * self.delay:
#             if self.free and avg_delay < self.switchpoints[0]:
#                 self.free = False
#                 print('switch free -> bump at t={}'.format(t))
#                 self.switch_times.append(t)
#                 self.t_delay = []
#                 self.x_delay = []
#             elif not self.free and avg_delay > self.switchpoints[1]:
#                 self.free = True
#                 print('switch bump -> free at t={}'.format(t))
#                 self.switch_times.append(t)
#                 self.t_delay = []
#                 self.x_delay = []
#         return out
#
#     def solve(self, t_0, x_0, t_1, t_eval=None):
#         """
#         return solution x(t_1) of IVP x(t_0) = x_0
#
#         Parameters
#         ----------
#         t_0 : float
#         x_0 : np.ndarray
#         t_1 : float
#         t_eval : np.ndarray, optional
#             intermediate values at which the state x is calculated and returned
#
#         Returns
#         -------
#         np.ndarray
#         """
#
#         if t_eval is None:
#             sol = solve_ivp(self, (t_0, t_1), x_0)
#             return sol.y[:, -1]
#
#         t_out = np.concatenate(([t_0], t_eval, [t_1]))
#         sol = solve_ivp(self, (t_0, t_1), x_0, t_eval=t_eval)
#         return sol.t, sol.y.T


class Pendulum(FunctionODE):
    """ A damped pendulum, which is bumped if the energy gets low  """

    def __init__(self, switchpoints=(-np.infty, np.infty)):
        """
        Parameters
        ----------
        switchpoints : Tuple[float], optional
            if bump and ||f|| > switchpoints[1]: switch to free
            if free and ||f|| < switchpoints[0]: switch to bump
        """
        super().__init__()
        self.switchpoints = switchpoints

        self.switch_times = []
        self.state_at_switch_times = []

        self.b = -0.2
        self.a = 2
        self.lamda = 5
        self.d = 1

    def reset(self, reset_params=True):
        self.evals = 0
        self.switch_times = []
        self.state_at_switch_times = []

    def __call__(self, t, x):
        """
        Parameters
        ----------
        t : float
        x : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self.evals += 1

        # calc switchpoints
        if len(self.switch_times) == 0:
            self.switch_times.append(t)
            self.state_at_switch_times.append(x)
            self.calc_switchtimes()
            self.calc_switchtimes()
        elif self.switch_times[-3] < t:
            self.calc_switchtimes()

        # find out if the system is free or bumped at time t
        this_free, idx = self.check_free(t)

        # calculate f(x, t)
        if this_free:
            out = np.array([[self.b, self.a], [-self.a, self.b]]) @ x
        else:
            out = np.array([0, self.lamda * (t - self.switch_times[idx]) + self.d])

        return out

    def solve(self, t_0, x_0, t_1, t_eval=None):
        """
        return solution x(t_1) of IVP x(t_0) = x_0

        Parameters
        ----------
        t_0 : float
        x_0 : np.ndarray
        t_1 : float
        t_eval : np.ndarray, optional
            intermediate values at which the state x is calculated and returned

        Returns
        -------
        np.ndarray
        """

        if t_eval is None:
            sol = solve_ivp(self, (t_0, t_1), x_0, max_step=0.1, rtol=1e-8)
            return sol.y[:, -1]

        t_out = np.concatenate(([t_0], t_eval, [t_1]))
        sol = solve_ivp(self, (t_0, t_1), x_0, t_eval=t_eval, rtol=1e-8)
        return sol.t, sol.y.T

    def calc_switchtimes(self):
        c, C = self.switchpoints
        t0 = self.switch_times[-1]
        x0 = self.state_at_switch_times[-1]

        t1 = np.log(c / np.linalg.norm([self.a, self.b]) / np.linalg.norm(x0)) / self.b + t0
        self.switch_times.append(t1)
        co = np.cos(self.a * (t1 - t0))
        si = np.sin(self.a * (t1 - t0))
        x1 = np.exp(self.b * (t1 - t0)) * np.array([[co, si], [-si, co]]) @ x0
        self.state_at_switch_times.append(x1)

        t2 = (C - self.d) / self.lamda + t1
        self.switch_times.append(t2)
        x2 = np.array([x1[0], self.lamda * 0.5 * (t2 - t1) ** 2 + self.d * (t2 - t1) + x1[1]])
        self.state_at_switch_times.append(x2)

    def check_free(self, t):
        last_idx = len(self.switch_times) - 1
        idx = last_idx
        while idx >= 0:
            if t >= self.switch_times[idx]:
                break
            idx -= 1

        if (last_idx - idx) % 2 == 0:
            return True, idx
        return False, idx


class LorenzSystem(FunctionODE):
    """ Lorenz system"""

    def __init__(self, switchpoints=(-np.infty, np.infty), chaotic=True, delay=0.1):
        """
        Parameters
        ----------
        switchpoints : Tuple[float], optional
            if chaotic and ||f|| > switchpoints[1]: switch to not chaotic
            if not chaotic and ||f|| < switchpoints[0]: switch to chaotic
        chaotic : bool, optional
            initial setting
        delay : float, optional
            the time that the system needs to exceed the switchpoint before a switch takes place
        """
        super().__init__()
        self.switchpoints = switchpoints
        self.delay = delay
        self.initial_chaotic = chaotic
        self.chaotic = chaotic

        self.sigma, self.beta, self.ro = self.choose_params()
        self.switch_times = []
        self.t_delay = []  # store times of delayed points
        self.x_delay = []  # store norm of delayed points

    def choose_params(self):
        sigma = 10
        beta = 8 / 3
        if self.chaotic:
            ro = 26
        else:
            ro = 9.5
        return sigma, beta, ro

    def reset(self, reset_params=True):
        self.evals = 0
        self.chaotic = self.initial_chaotic
        self.switch_times = []
        self.t_delay = []
        self.x_delay = []
        if reset_params:
            self.sigma, self.beta, self.ro = self.choose_params()

    def __call__(self, t, x):
        """
        f(t,x)=[sigma(x2-x1), x1(ro-x3)-x2, x1x2-betax3]

        Parameters
        ----------
        t : float
        x : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self.evals += 1
        out1 = self.sigma * (x[1] - x[0])
        out2 = x[0] * (self.ro - x[2]) - x[1]
        out3 = x[0] * x[1] - self.beta * x[2]
        out = np.array([out1, out2, out3])

        # calculate the average over delayed values
        abs_out = np.linalg.norm(out)
        self.t_delay.append(t)
        self.x_delay.append(abs_out)
        sorted_idx = sorted(range(len(self.t_delay)), key=self.t_delay.__getitem__)
        self.t_delay = [self.t_delay[idx] for idx in sorted_idx]
        self.x_delay = [self.x_delay[idx] for idx in sorted_idx]
        idx = 0
        while self.t_delay[-1] - self.t_delay[idx] > self.delay:
            idx += 1
        self.t_delay = self.t_delay[idx:]
        self.x_delay = self.x_delay[idx:]
        avg_delay = np.trapz(self.x_delay, self.t_delay)
        # print(self.t_delay[0], self.t_delay[-1], len(self.t_delay))
        # print(len(np.unique(self.t_delay)))
        # print(avg_delay)
        # print('')

        # switching
        if t >= self.t_delay[-1] and self.t_delay[-1] - self.t_delay[0] > 0.5 * self.delay:
            if self.chaotic and avg_delay > self.switchpoints[1]:
                self.chaotic = False
                self.sigma, self.beta, self.ro = self.choose_params()
                print('switch chaotic -> not chaotic at t={}'.format(t))
                self.switch_times.append(t)
                self.t_delay = []
                self.x_delay = []
            elif not self.chaotic and avg_delay < self.switchpoints[0]:
                self.chaotic = True
                self.sigma, self.beta, self.ro = self.choose_params()
                print('switch not chaotic -> chaotic at t={}'.format(t))
                self.switch_times.append(t)
                self.t_delay = []
                self.x_delay = []

        return out

    def solve(self, t_0, x_0, t_1, t_eval=None):
        """
        return solution x(t_1) of IVP x(t_0) = x_0

        Parameters
        ----------
        t_0 : float
        x_0 : np.ndarray
        t_1 : float
        t_eval : np.ndarray, optional
            intermediate values at which the state x is calculated and returned

        Returns
        -------
        np.ndarray
        """

        if t_eval is None:
            sol = solve_ivp(self, (t_0, t_1), x_0, rtol=1e-8)
            return sol.y[:, -1]

        t_out = np.concatenate(([t_0], t_eval, [t_1]))
        sol = solve_ivp(self, (t_0, t_1), x_0, t_eval=t_eval, rtol=1e-8)
        return sol.t, sol.y.T


class HenonHeiles(FunctionODE):
    def __init__(self):
        super().__init__()
        self.lamda = self.choose_params()

    @staticmethod
    def choose_params():
        return 1

    def reset(self, reset_params=True):
        self.evals = 0
        if reset_params:
            self.lamda = self.choose_params()

    def __call__(self, t, x):
        """
        Parameters
        ----------
        t : float
        x : np.ndarray
            shape=(4,), [x, p_x, y, p_y]

        Returns
        -------
        np.ndarray
        """
        self.evals += 1
        out = np.array([x[1],
                        -x[0] - 2 * self.lamda * x[0] * x[2],
                        x[3],
                        -x[2] - self.lamda * (x[0] ** 2 - x[2] ** 2)])
        return out

    def solve(self, t_0, x_0, t_1, t_eval=None):
        """
        return solution x(t_1) of IVP x(t_0) = x_0

        Parameters
        ----------
        t_0 : float
        x_0 : np.ndarray
        t_1 : float
        t_eval : np.ndarray, optional
            intermediate values at which the state x is calculated and returned

        Returns
        -------
        np.ndarray
        """

        if t_eval is None:
            sol = solve_ivp(self, (t_0, t_1), x_0, rtol=1e-8)
            return sol.y[:, -1]

        sol = solve_ivp(self, (t_0, t_1), x_0, t_eval=t_eval, rtol=1e-8)
        return sol.t, sol.y.T

    def calc_E(self, x):
        """Return the total energy of the system."""
        out = 0.5 * np.sum(x ** 2) + self.lamda * (x[0] ** 2 * x[2] - x[2] ** 3 / 3)
        return out

    def sample_initial_x(self):
        """ Uniformly sample x with energy 1/6. """
        x = np.zeros(4)

        # sample position
        a = np.array((0, 1))
        b = np.array((-0.5 * 3 ** 0.5, -0.5))
        c = np.array((0.5 * 3 ** 0.5, -0.5))
        r = np.random.random(2)
        x[[0, 2]] = (1 - r[0] ** 0.5) * a + r[0] ** 0.5 * (1 - r[1]) * b + r[0] ** 0.5 * r[1] * c
        pot_energy = 0.5 * np.sum(x ** 2) + self.lamda * (x[0] ** 2 * x[2] - x[2] ** 3 / 3)
        kin_energy = max(0.16666 - pot_energy, 0)

        # sample velocity
        v = np.random.normal(size=2)
        v = v / np.linalg.norm(v) * (2 * kin_energy) ** 0.5
        x[[1, 3]] = v

        return x


class VanDerPol(FunctionODE):
    def __init__(self):
        super().__init__()
        self.mu, self.amplitude, self.freq = self.choose_params()

    @staticmethod
    def choose_params():
        mu = 5
        amplitude = 5  # forcing
        frequency = 2.465  # frequency of forcing

        # mu = 5
        # amplitude = 50  # forcing
        # frequency = 7  # frequency of forcing
        return mu, amplitude, frequency

    def reset(self, reset_params=True):
        self.evals = 0
        if reset_params:
            self.mu, self.amplitude, self.freq = self.choose_params()

    def __call__(self, t, x):
        """
        Parameters
        ----------
        t : float
        x : np.ndarray
            shape=(2,)

        Returns
        -------
        np.ndarray
        """
        self.evals += 1
        out = np.array([x[1],
                        self.mu * (1 - x[0] ** 2) * x[1] - x[0] + self.amplitude * np.sin(self.freq * t)])
        return out

    def solve(self, t_0, x_0, t_1, t_eval=None):
        """
        return solution x(t_1) of IVP x(t_0) = x_0

        Parameters
        ----------
        t_0 : float
        x_0 : np.ndarray
        t_1 : float
        t_eval : np.ndarray, optional
            intermediate values at which the state x is calculated and returned

        Returns
        -------
        np.ndarray
        """
        if t_eval is None:
            sol = solve_ivp(self, (t_0, t_1), x_0, rtol=1e-8)
            return sol.y[:, -1]

        sol = solve_ivp(self, (t_0, t_1), x_0, t_eval=t_eval, rtol=1e-8)
        return sol.t, sol.y.T


class DoublePendulum(FunctionODE):
    def __init__(self):
        super().__init__()
        self.m, self.le, self.g = self.choose_params()

    @staticmethod
    def choose_params():
        m = (1, 1)  # mass
        le = (1, 1)  # length
        g = 10  # gravity
        return m, le, g

    def reset(self, reset_params=True):
        self.evals = 0
        if reset_params:
            self.m, self.le, self.g = self.choose_params()

    def __call__(self, t, x):
        """
        https://web.mit.edu/jorloff/www/chaosTalk/double-pendulum/double-pendulum-en.html

        Parameters
        ----------
        t : float
        x : np.ndarray
            shape=(4,), (phi_1, dt phi_1, phi_2, dt phi_2)

        Returns
        -------
        np.ndarray
        """
        self.evals += 1
        out = np.zeros(4)

        s, c = np.sin(x[0] - x[2]), np.cos(x[0] - x[2])
        out[0] = x[1]
        out[1] = self.m[0] * self.g * np.sin(x[2]) * c - self.m[1] * s * (self.le[0] * x[1] ** 2 * c +
                                                                          self.le[1] * x[3] ** 2) \
                 - np.sum(self.m) * self.g * np.sin(x[0])
        out[1] /= self.le[0] * (self.m[0] + self.m[1] * s ** 2)
        out[2] = x[3]
        out[3] = np.sum(self.m) * (self.le[0] * x[1] ** 2 * s - self.g * np.sin(x[2]) + self.g * np.sin(x[0]) * c) + \
                 self.m[1] * self.le[1] * x[3] ** 2 * s * c
        out[3] /= self.le[1] * (self.m[0] + self.m[1] * s ** 2)

        return out

    def solve(self, t_0, x_0, t_1, t_eval=None):
        """
        return solution x(t_1) of IVP x(t_0) = x_0

        Parameters
        ----------
        t_0 : float
        x_0 : np.ndarray
        t_1 : float
        t_eval : np.ndarray, optional
            intermediate values at which the state x is calculated and returned

        Returns
        -------
        np.ndarray
        """
        if t_eval is None:
            sol = solve_ivp(self, (t_0, t_1), x_0, max_step=0.01, rtol=1e-8)
            return sol.y[:, -1]

        sol = solve_ivp(self, (t_0, t_1), x_0, t_eval=t_eval, max_step=0.01, rtol=1e-8)
        return sol.t, sol.y.T

    def calc_E(self, x):
        """Return the total energy of the system."""
        m1, m2 = self.m
        L1, L2 = self.le
        g = self.g
        th1, th1d, th2, th2d = x.T
        V = -(m1 + m2) * L1 * g * np.cos(th1) - m2 * L2 * g * np.cos(th2)
        T = 0.5 * m1 * (L1 * th1d) ** 2 + 0.5 * m2 * ((L1 * th1d) ** 2 + (L2 * th2d) ** 2 +
                                                      2 * L1 * L2 * th1d * th2d * np.cos(th1 - th2))
        return T + V

    def sample_initial_x(self, energy=10):
        """ Sample x with certain energy. We set x[1]=x[3]=0, i.e., no inital velocity. """
        x = np.zeros(4)

        # try to sample x[2] such that a fitting x[0] exists
        for i in range(1000):
            theta2 = (2 * np.pi * np.random.random()) - np.pi
            cos_theta1 = energy + self.m[1] * self.g * self.le[1] * np.cos(theta2)
            cos_theta1 /= -np.sum(self.m) * self.g * self.le[0]
            if np.abs(cos_theta1) <= 1:
                theta1 = np.arccos(cos_theta1)
                if np.random.random() > 0.5:
                    theta1 *= -1
                x[0] = theta1
                x[2] = theta2
                return x

        raise RuntimeError(f"Cannot find initial condition with energy {energy}.")


def test_pendulum():
    switch = (0.05, 3.3)
    x0 = np.array([1, 1])
    t0 = 0
    t1 = 50
    tol = 0.00001
    pend = Pendulum(switch)
    sol = solve_ivp(pend, (t0, t1), x0, atol=tol, rtol=tol)
    print(pend.switch_times)

    t = sol.t
    x = sol.y

    deltas = [sol.t[idx + 1] - sol.t[idx] for idx in range(sol.t.shape[0] - 1)]
    deltas = [val for val in deltas for _ in range(2)]
    times_duplicates = [t[0]] + [val for val in t[1:-1] for _ in range(2)] + \
                       [t[-1]]

    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 10), dpi=150)
    color1 = 'k'
    axs[0].set_xlabel('t', color='k')
    axs[0].set_ylabel('x_1', color='k')
    axs[0].plot(t, x[0, :], '-', color=color1)
    axs[0].grid()

    axs[1].set_ylabel('x_2', color='k')
    axs[1].plot(t, x[1, :], '-', color=color1)
    axs[1].grid()

    axs[2].set_ylabel('step size ode45', color='k')
    axs[2].plot(times_duplicates, deltas, '-', color=color1)
    axs[2].grid()

    for xc in pend.switch_times[:-4]:
        axs[0].axvline(x=xc, color=color1, linestyle='--', linewidth=0.8)
        axs[1].axvline(x=xc, color=color1, linestyle='--', linewidth=0.8)
        axs[2].axvline(x=xc, color=color1, linestyle='--', linewidth=0.8)
    plt.show()


def test_Lorenz():
    switch = (0.1, 100)
    x0 = np.array([10, 10, 10])
    t0 = 0
    t1 = 15
    lor = LorenzSystem(switchpoints=switch, chaotic=True, delay=0.5)
    lor2 = LorenzSystem(switchpoints=switch, chaotic=True, delay=0.5)
    t_eval = np.linspace(t0, t1, 10000)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # _, y = lor.solve(t0, x0, t1, t_eval=t_eval)
    # print(lor.switch_times)
    # ax.plot(y[:, 0], y[:, 1], y[:, 2], 'b')
    # ax.scatter(y[-1, 0], y[-1, 1], y[-1, 2], c='r')

    # lor.reset()
    # y = lor.solve(t0, x0, t1)
    # print(lor.switch_times)
    # ax.scatter(y[0], y[1], y[2], c='k')
    #
    # plt.show()
    #
    lor.reset()
    sol = solve_ivp(lor, (t0, t1), x0)
    print(lor.switch_times)
    print(sol.nfev)
    print('')

    lor2.reset()
    sol2 = solve_ivp(lor2, (t0, t1), x0, t_eval=t_eval)
    print(lor2.switch_times)
    print(sol2.nfev)
    #
    # t = sol.t
    # x = sol.y
    # t2 = sol2.t
    # x2 = sol2.y
    #
    # deltas = [sol.t[idx + 1] - sol.t[idx] for idx in range(sol.t.shape[0] - 1)]
    # deltas = [val for val in deltas for _ in range(2)]
    # times_duplicates = [t[0]] + [val for val in t[1:-1] for _ in range(2)] + \
    #                    [t[-1]]
    #
    # deltas2 = [t2[idx + 1] - t2[idx] for idx in range(t2.shape[0] - 1)]
    # deltas2 = [val for val in deltas2 for _ in range(2)]
    # times_duplicates2 = [t2[0]] + [val for val in t2[1:-1] for _ in range(2)] + \
    #                     [t2[-1]]
    #
    # fig, axs = plt.subplots(4, sharex=True, figsize=(10, 10), dpi=150)
    # color1 = 'k'
    # color2 = 'r'
    # axs[0].set_xlabel('t', color='k')
    # axs[0].set_ylabel('x_1', color='k')
    # axs[0].plot(t, x[0, :], '-', color=color1)
    # axs[0].plot(t2, x2[0, :], '-', color=color2)
    # axs[0].grid()
    #
    # axs[1].set_ylabel('x_2', color='k')
    # axs[1].plot(t, x[1, :], '-', color=color1)
    # axs[1].plot(t2, x2[1, :], '-', color=color2)
    # axs[1].grid()
    #
    # axs[2].set_ylabel('x_3', color='k')
    # axs[2].plot(t, x[2, :], '-', color=color1)
    # axs[2].plot(t2, x2[2, :], '-', color=color2)
    # axs[2].grid()
    #
    # axs[3].set_ylabel('step size ode45', color='k')
    # axs[3].plot(times_duplicates, deltas, '-', color=color1)
    # axs[3].plot(times_duplicates2, deltas2, '-', color=color2)
    # axs[3].grid()
    #
    # for xc in lor.switch_times:
    #     axs[0].axvline(x=xc, color=color1, linestyle='--', linewidth=0.8)
    #     axs[1].axvline(x=xc, color=color1, linestyle='--', linewidth=0.8)
    #     axs[2].axvline(x=xc, color=color1, linestyle='--', linewidth=0.8)
    #     axs[3].axvline(x=xc, color=color1, linestyle='--', linewidth=0.8)
    # for xc in lor2.switch_times:
    #     axs[0].axvline(x=xc, color=color2, linestyle='--', linewidth=0.8)
    #     axs[1].axvline(x=xc, color=color2, linestyle='--', linewidth=0.8)
    #     axs[2].axvline(x=xc, color=color2, linestyle='--', linewidth=0.8)
    #     axs[3].axvline(x=xc, color=color2, linestyle='--', linewidth=0.8)
    #
    # plt.show()


def test_Rotation():
    rot = Rotation()
    x0 = np.array([1, 1])
    t0 = 0
    t1 = 20
    t_eval = np.linspace(t0, t1, 10000)

    tol = 0.0001

    sol = solve_ivp(rot, (t0, t1), x0, atol=tol, rtol=tol)
    print(sol.t.shape)
    for idx in range(len(sol.t) - 1):
        print(sol.t[idx + 1] - sol.t[idx])
    print(sol)

    sol2 = solve_ivp(rot, (t0, t1), x0, t_eval=t_eval)
    # plt.plot(sol2.t, sol2.y[0, :], 'k')
    # plt.plot(sol.t, sol.y[0, :], 'b')
    # plt.show()

    t = sol2.t
    x = sol2.y

    times_to_plot = sol.t
    nodes_to_plot = sol.y

    deltas = [sol.t[idx + 1] - sol.t[idx] for idx in range(sol.t.shape[0] - 1)]
    deltas = [val for val in deltas for _ in range(2)]

    times_duplicates = [times_to_plot[0]] + [val for val in times_to_plot[1:-1] for _ in range(2)] + \
                       [times_to_plot[-1]]

    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 10), dpi=150)
    color = 'tab:blue'
    axs[0].set_xlabel('t', color='k')
    axs[0].set_ylabel('x_1', color=color)
    axs[0].plot(t, x[0, :], '-', color='k')
    axs[0].plot(times_to_plot, nodes_to_plot[0, :], '-x', color=color)
    axs[0].grid()

    axs[1].set_ylabel('x_2', color=color)
    axs[1].plot(t, x[1, :], '-', color='k')
    axs[1].plot(times_to_plot, nodes_to_plot[1, :], '-x', color=color)
    axs[1].grid()

    color = 'tab:blue'
    axs[2].set_ylabel('step size', color=color)
    axs[2].plot(times_duplicates, deltas, 'x-', color=color)
    axs[2].grid()

    plt.show()


if __name__ == '__main__':
    # test_Lorenz()
    test_pendulum()
    # test_Rotation()

    # """ Small skript to visualize a function class and check if integrals are computed correctly """
    # f = BrokenPolynomial()
    # x = np.linspace(-1, 1, 1000)
    #
    # for i in range(10):
    #     f.reset()
    #     plt.plot(x, [f(num) for num in x])
    #     print(f.integral(-1, 1))
    #     print(quad(f, -1, 1)[0])
    #     print()
    #
    # plt.show()
