from copy import deepcopy
import numpy as np
from adaptive.environments import IntegrationEnv, ODEEnv
from adaptive.predictor import Predictor, PredictorODE
from adaptive.integrator import Integrator, IntegratorODE
from matplotlib import pyplot as plt
import joblib


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
            steps.append(env.evals / (env.timesteps[-1] - env.timesteps[0]))
            reward_stepwise /= steps[-1]
            rewards.append(reward_stepwise)

        this_error = np.mean(errors)
        this_num_evals = np.mean(steps)
        this_reward = np.mean(rewards)
        self.errors.append(this_error)
        self.num_evals.append(this_num_evals)
        self.rewards.append(this_reward)
        self.best_models.add_model(predictor.model.get_weights(), this_error, this_num_evals, this_reward)

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
        plt.scatter(x, y, c=(len(x) - 1) * ['b'] + ['g'])
        plt.yscale('log')
        plt.xscale('log')
        if len(self.errors) > 1:
            plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
        plt.xlabel('error')
        plt.ylabel('num_steps')
        plt.tight_layout()
        plt.grid(which='both')
        plt.savefig('performance_track_pareto.png')
        plt.close()

    def plot_best_models(self):
        errors = self.best_models.errors
        n_evals = self.best_models.num_evals
        rewards = self.best_models.rewards
        plt.scatter(errors, n_evals, c=rewards, cmap=plt.get_cmap("cool"))
        cbar = plt.colorbar()
        cbar.set_label('reward')
        plt.xlabel('error per step')
        plt.ylabel('feval per time')
        plt.grid()
        plt.tight_layout()
        plt.savefig('best_models.png')
        plt.close()


class BestPredictors:
    def __init__(self):
        self.weights = []
        self.errors = []
        self.num_evals = []
        self.rewards = []

    def add_model(self, weights, error, num_eval, reward):
        """
        Parameters
        ----------
        weights : np.ndarray
        error : float
        num_eval : float
        reward : float
        """
        if not self._is_dominated(error, num_eval):

            for i in reversed(range(len(self.errors))):
                if self._dominates(error, num_eval, self.errors[i], self.num_evals[i]):
                    self.errors.pop(i)
                    self.num_evals.pop(i)
                    self.weights.pop(i)
                    self.rewards.pop(i)

            self.weights.append(weights)
            self.errors.append(error)
            self.num_evals.append(num_eval)
            self.rewards.append(reward)

    def _is_dominated(self, error, num_eval):
        """
        Returns
        -------
        bool
            True, if (error, num_eval) is dominated by an existing point
        """
        for er, ev in zip(self.errors, self.num_evals):
            if self._dominates(er, ev, error, num_eval):
                return True
        return False

    @staticmethod
    def _dominates(er1, nev1, er2, nev2):
        if er2 >= er1 and nev2 >= nev1:
            return True
        return False

    def save(self, filename=None):
        if filename is None:
            filename = "best_models.pkl"
        joblib.dump(self.__dict__, open(filename, "wb"))

    def load(self, filename=None):
        if filename is None:
            filename = "best_models.pkl"
        loaded = joblib.load(open(filename, "rb"))
        self.__dict__ = loaded
        return self

    def best_by_reward(self):
        """
        Returns
        -------
        np.ndarray
            weights of model with the highest reward
        """
        idx = np.argmax(self.rewards)
        return deepcopy(self.weights[idx])
