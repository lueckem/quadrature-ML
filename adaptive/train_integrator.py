import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from functions import Sinus, SuperposeSinus, BrokenPolynomial, DoublePendulumInteg
from adaptive.environments import IntegrationEnv
from adaptive.integrator import Integrator, IntegratorLinReg, Simpson, Gauss21
from adaptive.predictor import Predictor, PredictorQ
from adaptive.build_models import build_value_model, build_estimator_model


def main():
    step_sizes = np.geomspace(0.1, 0.7, 20)
    x0 = 0
    x1 = 100
    integrator = Gauss21()
    dim_state = integrator.num_nodes + 1
    dim_action = len(step_sizes)
    memory = 0
    error_tol = 1e-7
    f = DoublePendulumInteg(x0, x1)
    scaler = load("scaler_integ.pkl")

    env = IntegrationEnv(fun=f, max_iterations=10000, initial_step_size=step_sizes[0],
                         error_tol=error_tol, nodes_per_integ=integrator.num_nodes, memory=memory,
                         x0=0, max_dist=x1 - x0, step_size_range=(step_sizes[0], step_sizes[-1]))
    predictor = PredictorQ(step_sizes=step_sizes,
                           model=build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor',
                                                   memory=memory),
                           scaler=scaler)

    train_models(env, predictor, integrator, 50)


def train_models(env, predictor, integrator, num_episodes):
    """
    Train linear regression model for each step size.

    The models are saved in a list under the name 'linreg_models.bin'.

    Parameters
    ----------
    env : IntegrationEnv
    predictor: PredictorQ
    integrator : Integrator
    num_episodes : int
    """

    models = []
    states, integrals = sample_trainingdata_from_predictor(env, predictor, integrator, num_episodes)
    dump([states, integrals], "training_data_weight.bin")
    # states, integrals = load("training_data_weight.bin")

    for step_size_idx in range(len(predictor.step_sizes)):
        print("{} samples for step_size {}".format(len(states[step_size_idx]), step_size_idx))
        step_size = predictor.step_sizes[step_size_idx]

        if len(states[step_size_idx]) < env.nodes_per_integ * 10:
            # there are not enough samples for a reliable calculation
            model = LinearRegression()
            weights = 0.5 * step_size * integrator.weights
            model.coef_ = weights
            model.intercept_ = np.zeros((1,))
        else:
            print(0.5 * step_size * integrator.weights)
            X = np.stack(states[step_size_idx])
            y = np.array(integrals[step_size_idx])
            model = train_and_slash(X, y, score_diff=1e-10)

        print(model.coef_)
        print(model.intercept_)
        print('')
        models.append(model)

    dump(models, 'linreg_integrator.bin')


def train_and_slash(X, y, score_diff=1e-8):
    model = LinearRegression().fit(X, y)
    score = model.score(X, y)
    print(f"score: {score}")

    while 1 - score > score_diff:
        # disregard the 2% of worst data and fit again
        pred = model.predict(X)
        error = np.abs(y - pred) ** 2
        ind_to_delete = error.argsort()[-(len(error) // 5):][::-1]
        X = np.delete(X, ind_to_delete, axis=0)
        y = np.delete(y, ind_to_delete)
        model = LinearRegression().fit(X, y)
        score = model.score(X, y)
        print(f"score: {score}")

    return model


def sample_trainingdata_from_predictor(env: IntegrationEnv, predictor: PredictorQ,
                                       integrator: Integrator, num_episodes: int):
    num_step_sizes = len(predictor.step_sizes)
    states = [[] for _ in range(num_step_sizes)]  # function evaluations
    integrals = [[] for _ in range(num_step_sizes)]  # integrals

    for episode in range(num_episodes):
        if episode % 2 == 0:
            print(episode)
        state = env.reset(integrator)
        done = False

        while not done:
            step_size = predictor(state)
            next_state, reward, done, info = env.iterate(step_size, integrator)

            f_evals = next_state[0][1:]
            step_size_idx = np.argwhere(np.isclose(predictor.step_sizes, step_size))[0, 0]
            states[step_size_idx].append(f_evals)
            integrals[step_size_idx].append(info["correct_integral"])
            state = next_state.copy()
    return states, integrals


if __name__ == '__main__':
    main()

    # combining data
    # states, integrals = load("training_data_weight.bin")
    # states2, integrals2 = load("training_data_weight2.bin")
    # states3 = [states[i] + states2[i] for i in range(len(states))]
    # integrals3 = [integrals[i] + integrals2[i] for i in range(len(integrals))]
    # dump([states3, integrals3], "training_data_weight3.bin")

