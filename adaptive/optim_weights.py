import joblib
import numpy as np
import math
from matplotlib import pyplot as plt
from adaptive.environments import ODEEnv
from adaptive.integrator import IntegratorODE, ClassicRungeKutta, RKDP
from functions import Rotation, LorenzSystem, Pendulum, VanDerPol, HenonHeiles, DoublePendulum
from adaptive.experience import ExperienceODE
from adaptive.predictor import PredictorODE, PredictorQODE, PredictorConstODE, MetaQODE
from adaptive.build_models import build_value_model, build_value_modelODE
from adaptive.performance_tracker import PerformanceTrackerODE, BestPredictors
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression


def preprocessing():
    """
    Given an already trained PredictorODE, sample data for integrator optimization.
    """
    system = DoublePendulum()
    max_dist = 200
    step_sizes = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
    dim_state = 6  # nodes per integration step
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    x0 = np.array([3, 0, 0, 0])
    d = x0.shape[0]  # dimension of the ODE state space
    error_tol = 0.0001

    scaler = load(open("test_scaler.pkl", "rb"))

    env = ODEEnv(fun=system, max_iterations=10000, initial_step_size=step_sizes[0],
                 step_size_range=(step_sizes[0], step_sizes[-1]), reward_fun=None,
                 error_tol=error_tol, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=max_dist)

    predictor = PredictorQODE(step_sizes=step_sizes,
                              model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
                                                         filename=None, lr=0.01, memory=memory),
                              scaler=scaler)

    best_predictors = BestPredictors().load()
    predictor.model.set_weights(best_predictors.best_by_reward())
    # predictor = PredictorConstODE(0.05)

    integrator = RKDP()

    data_stateODE = []  # save history of StateODE (h, k1, ..., k6)
    target = []  # save history of correct integrations

    # --------- integration loop -------------
    state = env.reset(reset_params=False, integrator=integrator)
    done = False
    while not done:
        step_size = predictor(state)
        node = env.nodes[-1]
        next_state, r, done, info = env.iterate(step_size, integrator)
        data_stateODE.append(next_state[0])
        target.append(info["correct_node"] - node)  # x_{n+1} - x_{n}

        state = next_state.copy()

    target = np.vstack(target)  # shape=(num_sample, d)

    data = np.zeros((len(data_stateODE), d, dim_state))
    for i, s_ode in enumerate(data_stateODE):
        k = s_ode.step_size * np.vstack(s_ode.f_evals)
        data[i, :, :] = k.T

    np.savez_compressed("optim_weights_data", x=data, y=target)


def optimize():
    data = np.load("optim_weights_data.npz")
    x, y = data["x"], data["y"]
    num_samples, d, _ = x.shape
    print(x.shape)
    print(y.shape)
    y = y.flatten()
    x = x.reshape((num_samples * d, -1))

    model = LinearRegression(fit_intercept=False).fit(x, y)
    print(model.coef_)
    print(model.intercept_)
    print(model.score(x, y))

    integrator = RKDP()
    integrator.b = model.coef_
    dump(integrator, "integrator_optim_weights.pkl")


if __name__ == '__main__':
    preprocessing()
    optimize()
