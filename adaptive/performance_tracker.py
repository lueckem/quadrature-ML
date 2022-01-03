from copy import deepcopy
import numpy as np
from adaptive.environments import IntegrationEnv, ODEEnv
from adaptive.predictor import Predictor, PredictorODE
from adaptive.integrator import Integrator, IntegratorODE
from matplotlib import pyplot as plt


class PerformanceTracker:
    def __init__(self, env, num_testfuns, x0, x1):
        """
        Create num_testfuns copies of the IntegrationEnv env with different functions.
        The testfunctions will be integrated from x0 to x1 and the performance will be saved.

        Parameters
        ----------
        env : IntegrationEnv
        num_testfuns : int
        x0 : float
        x1 : float
        """

        self.x0 = x0
        self.x1 = x1
        # create copies of the original environmen
        self.envs = [deepcopy(env) for _ in range(num_testfuns)]
        # change the functions
        for e in self.envs:
            e.reset(reset_params=True)
        # keep track of past errors and num_evals
        self.errors = []
        self.num_evals = []
        self.rewards = []

    def evaluate_performance(self, predictor, integrator):
        """
        Evaluate average performance of predictor on the saved test functions.

        Parameters
        ----------
        predictor : Predictor
        integrator : Integrator

        Returns
        -------
        tuple
            reward, error, num_evals
        """
        errors = []
        steps = []
        rewards = []
        for env in self.envs:
            state = env.reset(reset_params=False)
            env.x0 = self.x0
            reward_stepwise = 0
            while True:
                action = predictor(state)
                state, reward, _, _ = env.iterate(action, integrator)
                reward_stepwise += reward
                if env.nodes[-1] > self.x1:
                    break

            errors.append(np.mean(env.errors[2:]))
            steps.append(len(env.nodes))
            for idx, node in enumerate(env.nodes[-env.nodes_per_integ + 1:-1]):
                if node > self.x1:
                    steps[-1] += idx - (env.nodes_per_integ - 2)  # delete not needed evaluations
                    break
            reward_stepwise /= steps[-1]
            rewards.append(reward_stepwise)

        this_error = np.mean(errors)
        this_num_evals = np.mean(steps)
        this_reward = np.mean(rewards)
        self.errors.append(this_error)
        self.num_evals.append(this_num_evals)
        self.rewards.append(this_reward)

        return this_reward, this_error, this_num_evals

    def plot(self):
        plt.plot(self.rewards, 'b-x')
        plt.ylabel('reward per step')
        plt.grid()
        plt.savefig('performance_track_rewards.png')
        plt.close()

    def plot_pareto(self, num_points=0):
        x = np.array(self.errors)
        y = np.array(self.num_evals)
        if num_points > 0:
            x = x[-num_points:]
            y = y[-num_points:]
        plt.scatter(x, y)
        plt.yscale('log')
        plt.xscale('log')
        if len(self.errors) > 1:
            plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
        plt.xlabel('error')
        plt.ylabel('num_steps')
        plt.tight_layout()
        plt.grid(which='both')
        plt.savefig('performance_track_pareto.png', figsize=(6, 6), dpi=150)
        plt.close()


class PerformanceTrackerODE:
    def __init__(self, env, num_testfuns, t0, t1, integrator):
        """
        Create num_testfuns copies of the IntegrationEnv env with different functions.
        The testfunctions will be integrated from t0 to t1 and the performance will be saved.

        Parameters
        ----------
        env : ODEEnv
        num_testfuns : int
        t0 : float
        t1 : float
        integrator : IntegratorODE
        """

        self.t0 = t0
        self.t1 = t1
        # create copies of the original environmen
        self.envs = [deepcopy(env) for _ in range(num_testfuns)]
        # change the functions
        for e in self.envs:
            e.reset(reset_params=True, integrator=integrator)
        # keep track of past errors and num_evals
        self.errors = []
        self.num_evals = []
        self.rewards = []

        # keep track of pareto optimal models
        self.best_models = BestPredictors()

    def evaluate_performance(self, predictor, integrator):
        """
        Evaluate average performance of predictor on the saved test functions.

        Parameters
        ----------
        predictor : PredictorODE
        integrator : IntegratorODE

        Returns
        -------
        tuple
            reward, error, num_evals
        """
        errors = []
        steps = []
        rewards = []
        for env in self.envs:
            state = env.reset(reset_params=False, integrator=integrator)
            env.t0 = self.t0
            reward_stepwise = 0
            while True:
                action = predictor(state)
                state, reward, _, _ = env.iterate(action, integrator)
                reward_stepwise += reward
                if env.timesteps[-1] > self.t1:
                    break

            errors.append(np.mean(env.errors))
            steps.append(env.evals)
            reward_stepwise /= steps[-1]
            rewards.append(reward_stepwise)

        this_error = np.mean(errors)
        this_num_evals = np.mean(steps)
        this_reward = np.mean(rewards)
        self.errors.append(this_error)
        self.num_evals.append(this_num_evals)
        self.rewards.append(this_reward)
        self.best_models.add_model(predictor, this_error, this_num_evals)

        return this_reward, this_error, this_num_evals

    def plot(self):
        plt.plot(self.rewards, 'b-x')
        plt.ylabel('reward per step')
        plt.grid()
        plt.savefig('performance_track_rewards.pdf')
        plt.close()

    def plot_pareto(self, num_points=0):
        x = np.array(self.errors)
        y = np.array(self.num_evals)
        if num_points > 0:
            x = x[-num_points:]
            y = y[-num_points:]
        plt.scatter(x, y, c=(len(x) - 1) * ['b'] + ['g'])
        plt.yscale('log')
        plt.xscale('log')
        if len(self.errors) > 1:
            plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
        plt.xlabel('error')
        plt.ylabel('num_steps')
        plt.tight_layout()
        plt.grid(which='both')
        plt.savefig('performance_track_pareto.pdf')
        plt.close()

    def plot_best_models(self):
        errors = np.array(self.best_models.errors)
        n_evals = np.array(self.best_models.num_evals)
        idx_sort = np.argsort(errors)
        errors = errors[idx_sort]
        n_evals = n_evals[idx_sort]
        plt.plot(errors, n_evals, '--x')
        plt.xlabel('error')
        plt.ylabel('num_steps')
        plt.grid()
        plt.tight_layout()
        plt.savefig('best_models.pdf')
        plt.close()


class BestPredictors:
    def __init__(self):
        self.models = []
        self.errors = []
        self.num_evals = []

    def add_model(self, model, error, num_eval):
        """
        Parameters
        ----------
        model : PredictorODE
        error : float
        num_eval : float
        """
        if not self._is_dominated(error, num_eval):
            self.models.append(model)
            self.errors.append(error)
            self.num_evals.append(num_eval)

    def _is_dominated(self, error, num_eval):
        for er, ev in zip(self.errors, self.num_evals):
            if error >= er and num_eval >= ev:
                return True
        return False
