import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from functions import Sinus, SuperposeSinus, BrokenPolynomial
from adaptive.environments import IntegrationEnv
from adaptive.integrator import Integrator, IntegratorLinReg, Simpson
from adaptive.predictor import Predictor, PredictorQ
from adaptive.build_models import build_value_model
from adaptive.error_estimator import Estimator
from adaptive.build_models import build_estimator_model
from matplotlib import pyplot as plt


def main():
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    # step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
    dim_state = 3
    dim_action = len(step_sizes)
    memory = 1

    env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.075,
                         error_tol=7.5e-6, nodes_per_integ=dim_state, memory=memory,
                         x0=0, max_dist=20, step_size_range=(step_sizes[0], step_sizes[-1]))
    scaler = load('scaler.bin')
    predictor = PredictorQ(step_sizes=step_sizes,
                           model=build_value_model(dim_state=dim_state, dim_action=dim_action,
                                                   filename=None, lr=0.00001, memory=memory),
                           scaler=load('model_quad/model_sinus/Simpson/scaler.bin'))
    integrator = Simpson()

    estimator = Estimator(build_estimator_model(dim_state, lr=0.0001, filename='estimator'), scaler,
                          threshold=100 * 7.5e-6)

    train_model(estimator, env, predictor, integrator, 5000, scaler)


def train_model(estimator, env, predictor, integrator, num_episodes, scaler):
    """
    Train error estimator model .

    The model is saved under the name 'estimator'.

    Parameters
    ----------
    estimator : Estimator
    env : IntegrationEnv
    predictor: Predictor
    num_episodes : int
    integrator : Integrator
    scaler: StandardScaler
    """

    # train
    sample_trainingdata_from_predictor(env, predictor, integrator, num_episodes, threshold=estimator.threshold)
    states, errors = load("data_train_estimator")
    estimator.model.fit(scaler.transform(states), errors, epochs=100)
    estimator.model.save_weights('estimator')


def sample_trainingdata_from_predictor(env: IntegrationEnv, predictor: Predictor, integrator: Integrator,
                                       num_episodes: int, threshold: float):
    states = []
    errors = []

    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(episode)
        state = env.reset()
        done = False

        while not done:
            action = predictor(state)  # idx of step_size
            next_state, reward, done, info = env.iterate(action, integrator)
            states.append(next_state[:env.nodes_per_integ])
            errors.append(info["exact_error"])
            state = next_state

    states = np.stack(states)
    errors = np.array(errors)
    errors = np.reshape(errors, (-1, 1))

    states_large = states[np.squeeze(errors > threshold), :]
    states_small = states[np.squeeze(errors <= threshold), :]
    np.random.shuffle(states_large)
    np.random.shuffle(states_small)
    length = min(states_large.shape[0], states_small.shape[0])
    print('size of data set: {}'.format(2 * length))
    states_large = states_large[:length, :]
    states_small = states_small[:length, :]
    targets = np.zeros(2 * length)
    targets[:length] = np.ones(length)
    states = np.concatenate((states_large, states_small))
    mapIndexPosition = list(zip(states, targets))
    np.random.shuffle(mapIndexPosition)
    states, targets = zip(*mapIndexPosition)
    states = np.stack(states)
    targets = np.array(targets)

    dump((states, targets), "data_train_estimator")


def test_estimator_accuracy():
    """
    For the data acquired in "sample_trainingdata_from_predictor" we test in how many cases the estimator
    can successfully identify states with error higher than threshold, i.e. error > 3 * tol
    """
    dim_state = 3
    scaler = load('scaler.bin')
    estimator = Estimator(build_estimator_model(dim_state, lr=0.00001, filename='estimator'), scaler, 0.01)
    states, errors = load("data_train_estimator")

    correct = 0
    for state, error in zip(states, errors):
        est_error = estimator.is_error_large(state)
        if est_error == int(error):
            correct += 1
    print("accuracy: {}".format(correct / errors.shape[0]))


if __name__ == '__main__':
    main()
    # test_estimator_accuracy()