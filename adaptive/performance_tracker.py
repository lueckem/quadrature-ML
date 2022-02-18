from copy import deepcopy
import numpy as np
from adaptive.environments import IntegrationEnv, ODEEnv
from adaptive.predictor import Predictor, PredictorODE
from adaptive.integrator import Integrator, IntegratorODE
from matplotlib import pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression

from functions import HenonHeiles, DoublePendulum


class PerformanceTracker:
    def __init__(self, env, num_testfuns, x0, x1, integrator):
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
            e.reset(integrator, reset_params=True)
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
            state = env.reset(integrator, reset_params=False)
            env.x0 = self.x0
            reward_stepwise = 0
            while True:
                action = predictor(state)
                state, reward, _, _ = env.iterate(action, integrator)
                reward_stepwise += reward
                if env.nodes[-1] > self.x1:
                    break

            errors.append(np.mean(env.errors))
            steps.append(len(env.nodes))
            # for idx, node in enumerate(env.nodes[-env.nodes_per_integ + 1:-1]):
            #     if node > self.x1:
            #         steps[-1] += idx - (env.nodes_per_integ - 2)  # delete not needed evaluations
            #         break
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
    def __init__(self, env, num_testfuns, t0, t1, integrator, optimize_integrator=False):
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
        optimize_integrator : bool, optional
            If True, optimizes the integrator when evaluate_performance is called
        """

        self.t0 = t0
        self.t1 = t1
        # create copies of the original environmen
        self.envs = [deepcopy(env) for _ in range(num_testfuns)]
        # change the functions
        for e in self.envs:
            e.reset(reset_params=True, integrator=integrator)
        self.integrator = deepcopy(integrator)
        # keep track of past errors and num_evals
        self.errors = []
        self.num_evals = []
        self.rewards = []

        # keep track of pareto optimal models
        self.best_models = BestPredictors()

        self.optimize_integrator = optimize_integrator

    def evaluate_performance(self, predictor, nfev=0):
        """
        Evaluate average performance of predictor on the saved test functions.

        Parameters
        ----------
        predictor : PredictorODE
        nfev : int, optional
            number of function evaluations the model has trained on

        Returns
        -------
        tuple
            reward, error, num_evals
        """
        errors = []
        steps = []
        rewards = []

        # for optimizing integrator
        data_stateODE = []  # save history of StateODE (h, k1, ..., k6)
        target = []  # save history of correct integrations

        for env in self.envs:
            env.x0 = DoublePendulum().sample_initial_x(20)
            initial_energy = DoublePendulum().calc_E(env.x0)
            state = env.reset(reset_params=False, integrator=self.integrator)
            env.t0 = self.t0
            reward_stepwise = 0
            while True:
                energy = DoublePendulum().calc_E(env.nodes[-1])
                if energy / initial_energy > 1.3:
                    env.errors[-1] = 100 * env.error_tol
                    break
                node = env.nodes[-1]
                action = predictor(state)
                state, reward, _, info = env.iterate(action, self.integrator)
                data_stateODE.append(state[0])
                target.append(info["correct_node"] - node)  # x_{n+1} - x_{n}
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
        self.best_models.add_model(predictor.model.get_weights(), this_error, this_num_evals, this_reward,
                                   self.integrator.b, nfev=nfev)

        if self.optimize_integrator:
            dim_state = 6
            d = target[0].shape[0]
            target = np.vstack(target)
            data = np.zeros((len(data_stateODE), d, dim_state))
            for i, s_ode in enumerate(data_stateODE):
                k = s_ode.step_size * np.vstack(s_ode.f_evals)
                data[i, :, :] = k.T

            target = target.flatten()
            data = data.reshape((data.shape[0] * d, -1))
            model = LinearRegression(fit_intercept=False).fit(data, target)
            print(f"old weights: {self.integrator.b}")
            self.integrator.b = 0.95 * self.integrator.b + 0.05 * model.coef_
            print(f"new weights: {self.integrator.b}")

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
        self.weights_integrator = []
        self.nfev_training = []

    def add_model(self, weights, error, num_eval, reward, weights_integrator, nfev=0):
        """
        Parameters
        ----------
        weights : np.ndarray
        error : float
        num_eval : float
        reward : float
        weights_integrator : np.ndarray
        nfev : int, optional
            number of function evaluations the model has trained on
        """
        if not self._is_dominated(error, num_eval):

            for i in reversed(range(len(self.errors))):
                if self._dominates(error, num_eval, self.errors[i], self.num_evals[i]):
                    self.errors.pop(i)
                    self.num_evals.pop(i)
                    self.weights.pop(i)
                    self.rewards.pop(i)
                    self.weights_integrator.pop(i)
                    self.nfev_training.pop(i)

            print(f"BestPredictor: Added model with (err, nfev) = ({error},{num_eval}).")
            self.weights.append(weights)
            self.errors.append(error)
            self.num_evals.append(num_eval)
            self.rewards.append(reward)
            self.weights_integrator.append(weights_integrator)
            self.nfev_training.append(nfev)

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

    def best_integrator_by_reward(self):
        """
        Returns
        -------
        np.ndarray
            weights of integrator with the highest reward
        """
        idx = np.argmax(self.rewards)
        return deepcopy(self.weights_integrator[idx])

