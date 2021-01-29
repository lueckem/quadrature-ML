import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.externals.joblib import dump, load
from sklearn.preprocessing import StandardScaler
from functions import Sinus, SuperposeSinus, BrokenPolynomial
from adaptive.environments import IntegrationEnv
from adaptive.integrator import Integrator, IntegratorLinReg, Simpson
from adaptive.predictor import Predictor, PredictorQ
from adaptive.build_models import build_value_model, build_estimator_model
from adaptive.error_estimator import Estimator
from matplotlib import pyplot as plt


def main():
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    # step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
    dim_state = 3
    dim_action = len(step_sizes)
    memory = 0

    env = IntegrationEnv(fun=Sinus(), max_iterations=1000, initial_step_size=0.1,
                         step_sizes=step_sizes, error_tol=0.0005, memory=memory, nodes_per_integ=dim_state,
                         max_dist=20, x0=0)
    predictor = PredictorQ(
        build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor', memory=memory),
        load('scaler.bin'))

    # estimator = Estimator(build_estimator_model(dim_state, lr=0.0001, filename='estimator'), load('scaler.bin'),
    #                       threshold=100 * 7.5e-6)

    train_models(env, predictor, 1000)


def train_models(env, predictor, num_episodes, estimator=None):
    """
    Train linear regression model for each step size.

    The models are saved in a list under the name 'linreg_models.bin'.

    Parameters
    ----------
    env : IntegrationEnv
    predictor: Predictor
    num_episodes : int
    estimator : Estimator
    """

    models = []
    states, integrals = sample_trainingdata_from_predictor(env, predictor, num_episodes, estimator)

    for step_size_idx in range(len(env.step_sizes)):
        print("{} samples for step_size {}".format(len(states[step_size_idx]), step_size_idx))
        step_size = env.step_sizes[step_size_idx]

        if len(states[step_size_idx]) < 10:
            # there are not enough samples for a reliable calculation -> choose simpson
            model = LinearRegression()
            weights = 2 * step_size * np.array([1, 4, 1]) / 6
            model.coef_ = weights
            model.intercept_ = np.zeros((1,))
        else:
            X = np.stack(states[step_size_idx])
            y = np.array(integrals[step_size_idx])
            model = LinearRegression().fit(X, y)

            # disregard the 10% of worst data and fit again
            pred = model.predict(X)
            error = np.abs(y - pred)
            ind = error.argsort()[-(len(error) // 10):][::-1]
            print(X.shape)
            X = np.delete(X, ind, axis=0)
            print(X.shape)
            y = np.delete(y, ind)
            model = LinearRegression().fit(X, y)

            if np.linalg.norm(model.coef_) < 1e-4 * step_size:
                # weights are = 0 -> use Simpson
                weights = 2 * step_size * np.array([1, 4, 1]) / 6
                model.coef_ = weights
                model.intercept_ = np.zeros((1,))

        print(model.coef_)
        print(model.intercept_)
        print('')
        if (np.abs(model.coef_) > 10).any():
            model.coef_ = np.ones(len(model.coef_)) / len(model.coef_)
        models.append(model)

    dump(models, 'linreg_models_estim.bin', compress=True)


def sample_trainingdata_from_predictor(env: IntegrationEnv, predictor: Predictor, num_episodes: int, estimator=None):
    num_step_sizes = len(env.step_sizes)
    states = [[] for _ in range(num_step_sizes)]
    integrals = [[] for _ in range(num_step_sizes)]

    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(episode)
        state = env.reset()
        done = False

        while not done:
            action = predictor(state)  # idx of step_size
            next_state, reward, done, info = env.iterate(action, Integrator(), estimator=estimator)
            fa = info["f_a"]
            f_evals = next_state[1:env.nodes_per_integ]
            f_evals = np.concatenate(([fa], f_evals + fa))
            step_size_idx = info["step_size_idx"]
            states[step_size_idx].append(f_evals)
            integrals[step_size_idx].append(info["correct_integral"])
            state = next_state
    return states, integrals


if __name__ == '__main__':
    main()
