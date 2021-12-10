from copy import deepcopy

import numpy as np
import math
from matplotlib import pyplot as plt
from functions import Function, FunctionODE
from adaptive.integrator import Integrator, Simpson, IntegratorODE, StateODE
from adaptive.error_estimator import Estimator
from adaptive.reward_functions import Reward, RewardExp, RewardLog10


class IntegrationEnv:
    def __init__(self, fun, max_iterations, initial_step_size, error_tol,
                 x0=0.0, step_size_range=(0.01, 1), max_dist=np.infty, nodes_per_integ=3, memory=0,
                 reward_fun=None):
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
        nodes_per_integ : int, optional
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
        self._nodes = []  # contains copies of boundary nodes, for unique nodes use self.nodes
        self.errors = []
        self.deltas = []
        self.reps = 0  # counter to keep track of repititions

        self.error_tol = error_tol
        self.step_size_range = step_size_range

        if reward_fun is None:
            self.reward_fun = RewardLog10(error_tol, step_size_range, (0.1, 2))
        else:
            self.reward_fun = reward_fun

    def reset(self, reset_params=True, integrator=None):
        """
        Reset the environment and return the initial state.

        Parameters
        ----------
        reset_params : bool, optional
            if a different function should be created from the function class
        integrator : Integrator, optional
            for calculating errors in the initialization phase. If None, the errors are set to 0.

        Returns
        -------
        np.ndarray
            the initial state
        """

        self.fun.reset(reset_params)
        self.current_iteration = 0
        self.reps = 0

        # calculate nodes and error for first initialization step
        self._nodes = [self.x0 + k * self.initial_step_size for k in range(self.nodes_per_integ)]
        self.errors = []
        if integrator is None:
            self.errors.append(0)
        else:
            integ = integrator(np.array([self.initial_step_size] + self._calc_shifted_funvals(self._nodes)),
                               self.fun(self.x0))
            correct_integ = self.fun.integral(self._nodes[0], self._nodes[-1])
            self.errors.append(abs(integ - correct_integ))

        # calculate nodes and error for further initialization steps (if memory is > 0)
        for i in range(self.memory):
            new_nodes = [self._nodes[-1] + k * self.initial_step_size for k in range(self.nodes_per_integ)]
            self._nodes.extend(new_nodes)

            if integrator is None:
                self.errors.append(0)
            else:
                integ = integrator(np.array([self.initial_step_size] + self._calc_shifted_funvals(new_nodes)),
                                   self.fun(new_nodes[0]))
                correct_integ = self.fun.integral(new_nodes[0], new_nodes[-1])
                self.errors.append(abs(integ - correct_integ))

        self.deltas = [self.initial_step_size] * (self.memory + 1)

        next_state = [self.initial_step_size] + self._calc_shifted_funvals(self._nodes[-self.nodes_per_integ:])
        self.memory_states = []
        for i in range(self.memory):
            self.memory_states.append([self.initial_step_size] + self._calc_shifted_funvals(
                self._nodes[-(self.nodes_per_integ - 1) * (i + 2) - 1: -(self.nodes_per_integ - 1) * (i + 1)]
            ))
        next_state_mem = deepcopy(next_state)
        for s in self.memory_states:
            next_state_mem.extend(s)

        self.memory_states.insert(0, next_state)
        self.memory_states.pop()

        return np.asarray(next_state_mem, dtype='float32')

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
        next_state : np.ndarray
        reward : float
        done : bool
        info : dict
            additional information about the integration step:
            "correct_integral"
            "correct_integral_shifted"
            "exact_error"
            "f_a"
            "step_size"
        """
        info = {}

        a = self._nodes[-1]
        self.deltas.append(step_size)

        # calculate nodes in the current interval and build next_state
        next_nodes = [a + k * step_size for k in range(self.nodes_per_integ)]
        next_state = [step_size] + self._calc_shifted_funvals(next_nodes)

        # repeat step
        if step_size > self.step_size_range[0] and estimator is not None:
            if estimator.is_error_large(np.asarray(next_state)):
                step_size = self.step_size_range[0]
                next_nodes = [a + k * step_size for k in range(self.nodes_per_integ)]
                next_state = [step_size] + self._calc_shifted_funvals(next_nodes)
                self.reps += 1
        info["step_size"] = step_size

        # calculate reward
        b = next_nodes[-1]
        correct_integral = self.fun.integral(a, b)
        info["correct_integral"] = correct_integral
        info["correct_integral_shifted"] = correct_integral - self.fun(a) * (self.nodes_per_integ - 1) * step_size
        integral = integrator(np.asarray(next_state, dtype='float32'), self.fun(a))
        error = abs(correct_integral - integral)
        info["exact_error"] = error
        info["f_a"] = self.fun(a)

        self.errors.append(error)
        reward = self.reward_fun(error, step_size)

        self._nodes = self._nodes + next_nodes

        # include the memory in next_state:
        next_state_mem = deepcopy(next_state)
        for s in self.memory_states:
            next_state_mem.extend(s)
        self.memory_states.insert(0, next_state)
        self.memory_states.pop()

        next_state_mem = np.asarray(next_state_mem, dtype='float32')
        self.current_iteration += 1
        if self.current_iteration >= self.max_iterations or b >= self.x0 + self.max_dist:
            done = True
        else:
            done = False

        return next_state_mem, reward, done, info

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
        """

        if x_min is None:
            id_min = 0
            x_min = self._nodes[0]
        else:
            id_min = next(i for i, node in enumerate(self._nodes) if node >= x_min)

        if x_max is None:
            id_max = len(self._nodes) - 1
            x_max = self._nodes[-1]
        elif x_max < self._nodes[-1]:
            id_max = next(i for i, node in enumerate(self._nodes) if node > x_max)
        else:
            id_max = len(self._nodes)

        nodes_to_plot = self._nodes[id_min:id_max + 1]
        nodes_to_plot_unique = np.unique(nodes_to_plot)
        num_steps = math.ceil((x_max - x_min) / (self.initial_step_size / 8.0))
        x = np.linspace(nodes_to_plot[0], nodes_to_plot[-1], num_steps)
        y = [self.fun(num) for num in x]
        y_nodes = [self.fun(num) for num in nodes_to_plot_unique]

        errors = [val for val in self.errors for _ in range(self.nodes_per_integ)]
        errors = errors[id_min:id_max + 1]
        deltas = [val for val in self.deltas for _ in range(self.nodes_per_integ)]
        deltas = deltas[id_min:id_max + 1]

        # plt.rcParams.update({'font.size': 14})
        # fig, axs = plt.subplots(3, sharex=True, figsize=(7, 6), dpi=300)
        fig, axs = plt.subplots(3, sharex=True)
        color = 'tab:blue'
        axs[0].set_xlabel('x', color='k')
        axs[0].set_ylabel('f(x)', color=color)
        axs[0].plot(x, y, color=color)
        axs[0].tick_params(axis='y', labelcolor='k')
        color = 'tab:green'
        axs[0].plot(nodes_to_plot_unique, y_nodes, 'x', color=color)
        axs[0].plot(nodes_to_plot_unique, np.zeros((len(nodes_to_plot_unique),)), '|', color=color)
        axs[0].grid()

        color = 'tab:red'
        axs[1].set_ylabel('error', color=color)
        axs[1].plot(nodes_to_plot, errors, 'x-', color=color)
        axs[1].plot(nodes_to_plot, self.error_tol * np.ones(len(nodes_to_plot)), 'k-')
        axs[1].grid()

        color = 'tab:blue'
        axs[2].set_ylabel('step size', color=color)
        axs[2].plot(nodes_to_plot, deltas, 'x-', color=color)
        axs[2].grid()

        fig.tight_layout()
        if save:
            plt.savefig('adapt_{}.png'.format(episode))
        if show:
            plt.show()
        plt.close()

    def sample_states(self, num_samples):
        """
        Uniformly samples the state space for creation of a scaler.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        np.ndarray
            samples, shape (num_samples, self.nodes_per_integ * (memory + 1)
        """
        samples = np.zeros((num_samples, self.nodes_per_integ))
        samples[:, 0] = (self.step_size_range[1] - self.step_size_range[0]) * np.random.sample(num_samples)\
                        + self.step_size_range[0]
        max_dif = self.fun.maximum() - self.fun.minimum()
        samples[:, 1:] = 2 * max_dif * np.random.sample((num_samples, self.nodes_per_integ - 1)) - max_dif
        samples = np.tile(samples, (1, self.memory + 1))
        print(samples.shape)
        return samples

    @property
    def nodes(self):
        nodes = np.array(self._nodes)
        return np.unique(nodes)

    @property
    def evals(self):
        return len(self.nodes) + (self.nodes_per_integ - 1) * self.reps

    def _calc_shifted_funvals(self, nodes):
        """
        For a list of nodes, calculate the function values of (f-f(x_0)), so that the first value is centered to 0.
            In the output the 0 is left out, so the length of the output is one less than of nodes.
            nodes has to be in ascending order!
        """

        out = []
        shift = self.fun(nodes[0])
        for node in nodes[1:]:
            out.append(self.fun(node) - shift)
        return out


class ODEEnv:
    def __init__(self, fun, max_iterations, initial_step_size, error_tol, x0, t0=0, step_size_range=(0.01, 10),
                 max_dist=np.infty, nodes_per_integ=4, memory=0, reward_fun=None, stepsize_to_idx=None):
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
        if self.current_iteration >= self.max_iterations or next_time >= self.t0 + self.max_dist:
            done = True
        else:
            done = False

        return self.memory_states.copy(), reward, done, info

    def plot(self, t_min=None, t_max=None, episode=0, show=True, save=False):
        """
        plot the first 2 dimensions of x, the errors, and the step_sizes against the time

        Parameters
        ----------
        t_min : float
            left bound of x-axis
        t_max : float
            right bound of x-axis
        episode : int
            for labeling the file
        """

        if t_min is None:
            id_min = 0
            t_min = self.timesteps[0]
        else:
            id_min = next(i for i, t in enumerate(self.timesteps) if t >= t_min)

        if t_max is None:
            id_max = len(self.timesteps) - 1
            t_max = self.timesteps[-1]
        elif t_max < self.timesteps[-1]:
            id_max = next(i for i, t in enumerate(self.timesteps) if t > t_max) - 1
        else:
            id_max = len(self.timesteps) - 1

        times_to_plot = self.timesteps[id_min:id_max + 1]
        nodes_to_plot = np.array(self.nodes[id_min:id_max + 1])

        # num_steps = math.ceil((t_max - t_min) / (self.initial_step_size / 8.0))
        # t = np.linspace(times_to_plot[0], times_to_plot[-1], num_steps)
        # if id_min == 0:
        #     t, x = self.fun.solve(self.t0, self.x0, t[-1], t_eval=t[1:-1])
        # else:
        #     t, x = self.fun.solve(self.t0, self.x0, t[-1], t_eval=t[:-1])

        errors = self.errors[id_min:id_max]
        errors = [val for val in errors for _ in range(2)]

        deltas = self.deltas[id_min:id_max]
        deltas = [val for val in deltas for _ in range(2)]

        times_duplicates = [times_to_plot[0]] + [val for val in times_to_plot[1:-1] for _ in range(2)] +\
                           [times_to_plot[-1]]

        dim = self.x0.shape[0]
        fig, axs = plt.subplots(dim + 2, sharex=True)
        color = 'tab:blue'
        for i in range(dim):
            axs[i].set_ylabel(r'$x_{}$'.format(i + 1), color=color)
            axs[i].plot(times_to_plot, nodes_to_plot[:, i], '-x', color=color)
            axs[i].grid()

        color = 'tab:red'
        axs[dim].set_ylabel('error', color=color)
        axs[dim].plot(times_duplicates, errors, 'x-', color=color)
        axs[dim].plot(times_duplicates, self.error_tol * np.ones(len(times_duplicates)), 'k-')
        axs[dim].grid()

        color = 'tab:blue'
        axs[dim + 1].set_ylabel('step size', color=color)
        axs[dim + 1].plot(times_duplicates, deltas, 'x-', color=color)
        axs[dim + 1].grid()

        fig.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        if save:
            plt.savefig('adapt_{}.png'.format(episode))
        if show:
            plt.show()
        plt.close()

    @property
    def evals(self):
        return (len(self.nodes) - 1) * self.nodes_per_integ + self.nodes_per_integ * self.reps
