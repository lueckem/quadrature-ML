import math
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from adaptive.error_estimator import Estimator
from adaptive.integrator import Integrator, IntegratorODE, Simpson, StateODE
from adaptive.plots import plot_trajectory, plot_trajectory_quad
from adaptive.reward_functions import Reward, RewardExp, RewardLog10
from functions import Function, FunctionODE


class IntegrationEnv:
    def __init__(
        self,
        fun,
        max_iterations,
        initial_step_size,
        error_tol,
        nodes_per_integ,
        x0=0.0,
        step_size_range=(0.01, 1),
        max_dist=np.infty,
        memory=0,
        reward_fun=None,
    ):
        """
        Parameters
        ----------
        fun : Function
            Function class that is to be integrated
        max_iterations : int
            number of iterations per episode
        max_dist : float
            maximum distance to integrate from x0 to
        initial_step_size : float
        error_tol : float
            if the error is larger than the error tolerance, negative rewards are given
        x0 : float, optional
            starting node
        step_size_range : tuple[float]
            (smallest step size, largest step size), the positive rewards are chosen accordingly
        nodes_per_integ : int
            how many nodes are used for one iteration
        memory : int, optional
            how many iterations in the past are saved in the state
            default: 0
            if > 0, the states are simply concatenated from present to past
        reward_fun : Reward, optional
            reward function, chooses RewardLog10 per default
        """

        self.x0 = x0
        self.fun = fun
        self.max_iterations = max_iterations
        self.max_dist = max_dist
        self.initial_step_size = initial_step_size
        self.nodes_per_integ = nodes_per_integ
        self.memory = memory
        self.memory_states = []

        self.current_iteration = 0
        self.nodes = []  # start and end points of subintervals
        self.errors = []  # step-wise errors
        self.deltas = []  # chosen step sizes
        self.integrals = []  # integrals of steps
        self.reps = 0  # counter to keep track of repititions

        self.error_tol = error_tol
        self.step_size_range = step_size_range

        if reward_fun is None:
            self.reward_fun = RewardLog10(error_tol, step_size_range, (0.1, 2))
        else:
            self.reward_fun = reward_fun

    def reset(self, integrator, reset_params=True):
        """
        Reset the environment and return the initial state.

        Parameters
        ----------
        integrator : Integrator
            for calculating errors in the initialization phase. If None, the errors are set to 0.
        reset_params : bool, optional
            if a different function should be created from the function class

        Returns
        -------
        list[np.ndarray]
            the initial state
        """

        self.fun.reset(reset_params)
        self.current_iteration = 0
        self.reps = 0

        node = self.x0
        h = self.initial_step_size

        self.nodes = [node]
        self.deltas = []
        self.errors = []
        self.integrals = []
        self.memory_states = []

        for i in range(self.memory + 1):
            self.deltas.append(h)
            state = integrator.calc_state(node, h, self.fun)
            self.memory_states.append(state)
            integ = integrator(state)
            self.integrals.append(integ)
            correct_integ = self.fun.integral(node, node + h)
            self.errors.append(abs(integ - correct_integ))

            node = node + h
            self.nodes.append(node)

        self.memory_states.reverse()
        return self.memory_states.copy()

    def iterate(self, step_size, integrator, estimator=None):
        """
        Iterate the environment one step forward.

        Parameters
        ----------
        step_size : float
            step size for this iteration
        integrator : Integrator
        estimator : Estimator
            if an estimator is given, the current step is being repeated with the smallest step size if the error is
            estimated to be higher than the threshold and the step size is not already small

        Returns
        -------
        next_state : list[np.ndarray]
        reward : float
        done : bool
        info : dict
            additional information about the integration step:
            "correct_integral"
            "exact_error"
        """
        info = {}

        node = self.nodes[-1]
        if node + step_size > self.nodes[0] + self.max_dist:
            step_size = max(self.nodes[0] + self.max_dist - node, 1e-8)
        self.deltas.append(step_size)

        state = integrator.calc_state(node, step_size, self.fun)
        integral = integrator(state)

        # repeat step
        # if step_size > self.step_size_range[0] and estimator is not None:
        #     if estimator.is_error_large(np.asarray(next_state)):
        #         step_size = self.step_size_range[0]
        #         next_nodes = [a + k * step_size for k in range(self.nodes_per_integ)]
        #         next_state = [step_size] + self._calc_shifted_funvals(next_nodes)
        #         self.reps += 1

        # calculate reward
        correct_integral = self.fun.integral(node, node + step_size)
        info["correct_integral"] = correct_integral
        error = abs(correct_integral - integral)
        info["exact_error"] = error

        self.errors.append(error)
        self.integrals.append(integral)
        reward = self.reward_fun(error, step_size)

        next_node = node + step_size
        self.nodes.append(next_node)

        self.memory_states.insert(0, state)
        self.memory_states.pop()

        self.current_iteration += 1
        if (
            self.current_iteration >= self.max_iterations
            or next_node >= self.x0 + self.max_dist
        ):
            done = True
        else:
            done = False

        return self.memory_states.copy(), reward, done, info

    def plot(self, x_min=None, x_max=None, episode=0, save=False, show=True):
        """
        Parameters
        ----------
        x_min : float
            left bound of x-axis
        x_max : float
            right bound of x-axis
        episode : int
            for labeling the file
        save : bool, optional
        show : bool, optional
        """

        fig, ax = plot_trajectory_quad(
            nodes=self.nodes,
            f=self.fun,
            errors=self.errors,
            deltas=self.deltas,
            error_tolerance=self.error_tol
        )

        if save:
            fig.savefig("quad_{}.pdf".format(episode))
        if show:
            plt.show()
        plt.close()

    # def sample_states(self, num_samples):
    #     """
    #     Uniformly samples the state space for creation of a scaler.
    #
    #     Parameters
    #     ----------
    #     num_samples : int
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         samples, shape (num_samples, self.nodes_per_integ * (memory + 1)
    #     """
    #     samples = np.zeros((num_samples, self.nodes_per_integ))
    #     samples[:, 0] = (
    #         self.step_size_range[1] - self.step_size_range[0]
    #     ) * np.random.sample(num_samples) + self.step_size_range[0]
    #     max_dif = self.fun.maximum() - self.fun.minimum()
    #     samples[:, 1:] = (
    #         2 * max_dif * np.random.sample((num_samples, self.nodes_per_integ - 1))
    #         - max_dif
    #     )
    #     samples = np.tile(samples, (1, self.memory + 1))
    #     print(samples.shape)
    #     return samples

    @property
    def evals(self):
        """
        Number function evaluations used.

        If the integrator uses the boundary nodes of subintervals, the output is not correct
        because the boundary nodes are counted twice.

        Returns
        -------
            float
        """
        num_subintervals = len(self.nodes) - 1
        return num_subintervals * self.nodes_per_integ

    @property
    def integral(self):
        return np.sum(self.integrals)


class ODEEnv:
    def __init__(
        self,
        fun,
        max_iterations,
        initial_step_size,
        error_tol,
        x0,
        t0=0,
        step_size_range=(0.01, 10),
        max_dist=np.infty,
        nodes_per_integ=4,
        memory=0,
        reward_fun=None,
        stepsize_to_idx=None,
    ):
        """
        Parameters
        ----------
        fun : FunctionODE
            Function class that is to be integrated
        max_iterations : int
            number of iterations per episode
        max_dist : float
            maximum time to integrate from t0 to
        initial_step_size : float
        step_size_range : tuple[float]
            (smallest step size, largest step size), the positive rewards are chosen accordingly
        error_tol : float
            if the error is larger than the error tolerance, negative rewards are given
        x0 : np.ndarray
            starting node
        t0 : float
            starting time
        nodes_per_integ : int, optional
            how many evals f(t,x) are used for one iteration
        memory : int, optional
            how many iterations in the past are saved in the state
            default: 0
            if > 0, the states are simply concatenated from present to past
        reward_fun : Reward, optional
            reward function, chooses RewardLog10 per default
        stepsize_to_idx : function, optional
            convert step_size in state to idx, in order to fill the value StateODE.step_idx
        """

        self.t0 = t0
        self.x0 = x0
        self.fun = fun
        self.max_iterations = max_iterations
        self.max_dist = max_dist
        self.initial_step_size = initial_step_size
        self.nodes_per_integ = nodes_per_integ
        self.memory = memory
        self.memory_states = []

        self.current_iteration = 0
        self.nodes = []  # [x(t_0), x(t_1), ...]
        self.timesteps = []  # [t_0, t_1, ...]
        self.errors = []
        self.deltas = []  # [t_1 - t_0, t_2 - t_1, ...]
        self.reps = 0  # counter to keep track of repititions

        self.error_tol = error_tol
        self.step_size_range = step_size_range

        if reward_fun is None:
            self.reward_fun = RewardLog10(error_tol, step_size_range, (0.1, 2))
        else:
            self.reward_fun = reward_fun

        self.stepsize_to_idx = stepsize_to_idx

    def reset(self, integrator, reset_params=True):
        """
        Reset the environment and return the initial state.

        Parameters
        ----------
        integrator : IntegratorODE
            for calculating errors in the initialization phase. If None, the errors are set to 0.
        reset_params : bool, optional
            if a different function should be created from the function class

        Returns
        -------
        list[StateODE]
            the initial state
        """

        self.fun.reset(reset_params)
        self.current_iteration = 0
        self.reps = 0

        node = self.x0
        time = self.t0

        self.nodes = [node]
        self.timesteps = [time]
        self.deltas = []
        self.errors = []
        self.memory_states = []
        for i in range(self.memory + 1):
            self.deltas.append(self.initial_step_size)
            correct_node = self.fun.solve(time, node, time + self.initial_step_size)
            state = integrator.calc_state(time, node, self.initial_step_size, self.fun)
            if self.stepsize_to_idx is not None:
                state.step_idx = self.stepsize_to_idx(self.initial_step_size)
            self.memory_states.append(state)
            node = integrator(state, node)
            time = time + self.initial_step_size
            self.errors.append(np.linalg.norm(node - correct_node))
            self.nodes.append(node)
            self.timesteps.append(time)
        self.memory_states.reverse()

        return self.memory_states.copy()

    def iterate(self, step_size, integrator, estimator=None):
        """
        Iterate the environment one step forward.

        Parameters
        ----------
        step_size : float
            step size for this iteration
        integrator : IntegratorODE
        estimator : Estimator
            if an estimator is given, the current step is being repeated with the smallest step size if the error is
            estimated to be higher than the threshold and the step size is not already small

        Returns
        -------
        next_state : list[StateODE]
        reward : float
        done : bool
        info : dict
            additional information about the integration step:
            "correct_node"
            "exact_error"
        """
        info = {}

        node = self.nodes[-1]
        time = self.timesteps[-1]
        self.deltas.append(step_size)

        state = integrator.calc_state(time, node, step_size, self.fun)
        if self.stepsize_to_idx is not None:
            state.step_idx = self.stepsize_to_idx(step_size)
        next_node = integrator(state, node)
        next_time = time + step_size

        # repeat step
        # if step_size_idx > 1 and estimator is not None:
        #     if estimator.is_error_large(np.asarray(next_state)):
        #         step_size = self.step_sizes[0]
        #         step_size_idx = 0
        #         next_nodes = [a + k * step_size for k in range(self.nodes_per_integ)]
        #         next_state = [step_size] + self._calc_shifted_funvals(next_nodes)
        #         self.reps += 1

        # calculate reward
        correct_node = self.fun.solve(time, node, next_time)
        # print(correct_node)
        info["correct_node"] = correct_node
        error = np.linalg.norm(next_node - correct_node)
        info["exact_error"] = error
        self.errors.append(error)
        reward = self.reward_fun(error, step_size)

        self.nodes.append(next_node)
        self.timesteps.append(next_time)

        # include the memory
        self.memory_states.insert(0, state)
        self.memory_states.pop()

        self.current_iteration += 1
        if (
            self.current_iteration >= self.max_iterations
            or next_time >= self.t0 + self.max_dist
        ):
            done = True
        else:
            done = False

        return self.memory_states.copy(), reward, done, info

    def plot(
        self,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        episode: int = 0,
        show: bool = True,
        save: bool = False,
    ):
        """Plot the first 2 dimensions of x, the errors, and the step_sizes against time

        Parameters
        ----------
        t_min : float, optional
            Left bound of x-axis. If None, the left bound is set to the first timestep.
        t_max : float
            Right bound of x-axis. If None, the right bound is set to the last timestep.
        episode : int
            Index of the episode to plot. Will be used for the filename if save is True.
        show : bool
            If True, the plot will be shown.
        save : bool
            If True, the plot will be saved to a file.
        """

        fig, ax = plot_trajectory(
            x0=self.x0,
            timesteps=self.timesteps,
            nodes=self.nodes,
            errors=self.errors,
            deltas=self.deltas,
            t_min=t_min,
            t_max=t_max,
            error_tolerance=self.error_tol,
        )

        if save:
            fig.savefig("adapt_{}.pdf".format(episode))
        if show:
            plt.show()
        plt.close()

    @property
    def evals(self):
        return (
            len(self.nodes) - 1
        ) * self.nodes_per_integ + self.nodes_per_integ * self.reps
