import numpy as np
from adaptive.environments import IntegrationEnv
from adaptive.build_models import build_value_model
from functions import Sinus, SuperposeSinus
from adaptive.integrator import Integrator, Simpson, IntegratorLinReg
from joblib import dump, load
from copy import deepcopy
from adaptive.predictor import *


def benchmark(predictors, integrators, num_episodes, env):
    """
    Benchmark a list of predictors.

    Returns the cumulative reward of each predictor integrating the given environment num_episode times.

    Parameters
    ----------
    predictors : list[Predictor]
    integrators : Integrator or list[Integrator]
    num_episodes : int
    env : IntegrationEnv

    Returns
    -------
    np.ndarray
        rewards of each predictor
    """

    num_pred = len(predictors)
    scores = np.zeros((num_pred,))

    if isinstance(integrators, Integrator):
        integrators = [integrators] * num_pred

    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(episode)
        env.reset()

        for idx, predictor in enumerate(predictors):
            state = env.reset(reset_params=False)
            done = False
            while not done:
                action = predictor(state)
                next_state, reward, done, _ = env.iterate(action, integrators[idx])
                scores[idx] += reward
                state = next_state

    return scores


def main():
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    dim_state = 3
    dim_action = len(step_sizes)
    env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.1,
                         step_sizes=step_sizes, error_tol=0.0005, nodes_per_integ=dim_state)
    num_episodes = 500

    predictors = [PredictorConst(i) for i in range(dim_action)]
    predictors.append(PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
                                 load('scaler.bin')))

    scores = benchmark(predictors,
                       IntegratorLinReg(step_sizes, load('linreg_models.bin'), load('scaler.bin')),
                       num_episodes,
                       env)

    # scores = benchmark(predictors,
    #                    Simpson(),
    #                    num_episodes,
    #                    env)

    print(scores / num_episodes)


if __name__ == '__main__':
    main()
