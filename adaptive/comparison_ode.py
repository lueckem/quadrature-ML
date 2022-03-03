import numpy as np
from matplotlib import pyplot as plt
from adaptive.environments import ODEEnv
from adaptive.integrator import IntegratorODE, ClassicRungeKutta, RKDP
from functions import Rotation, LorenzSystem, Pendulum, VanDerPol, HenonHeiles, DoublePendulum
from adaptive.predictor import PredictorODE, PredictorQODE, PredictorConstODE, MetaQODE
from adaptive.build_models import build_value_model, build_value_modelODE
from adaptive.performance_tracker import PerformanceTrackerODE, BestPredictors
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
from adaptive.plots import plot_pareto, plot_trajectory

from adaptive.train_metalearner import stepsize_to_idx


def integrate_env(predictor, integrator, env, t0=None, x0=None, t1=None, plot=False):
    """
    Integrates the environment from t0 to t1 using the specified predictor and integrator.

    Parameters
    ----------
    predictor : PredictorODE
    integrator : IntegratorODE
    env : ODEEnv
        does not change the function
    t0 : float, optional
    x0 : np.ndarray, optional
    t1 : float, optional
    plot : bool, optional

    Returns
    -------
    reward : float
    num_evals : int
        number of function evaluations
    """
    if t0 is not None:
        env.t0 = t0
    if x0 is not None:
        env.x0 = x0
    if t1 is None:
        if env.max_dist < np.infty:
            t1 = env.t0 + env.max_dist
        else:
            raise ValueError("Give value for t1!")

    state = env.reset(reset_params=False, integrator=integrator)
    reward = 0
    # energy = []
    # initial_energy = DoublePendulum().calc_E(env.x0)
    while True:
        # energy.append(env.fun.calc_E(env.nodes[-1]))
        # energy = DoublePendulum().calc_E(env.nodes[-1])
        # if energy / initial_energy > 1.3:
        #     break
        step_size = predictor(state)
        next_state, r, _, info = env.iterate(step_size, integrator)
        reward += r
        state = next_state.copy()
        if env.timesteps[-1] >= t1:
            break

    num_evals = env.evals

    if plot:
        env.plot(save=True)

    return reward, num_evals


def onefun():
    """
    Integrate and plot one ODE (using PredictorODE) and print important statistics.
    """
    # step_sizes = [0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070]
    step_sizes = [0.56, 0.58, 0.6, 0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.8]
    dim_state = 6  # nodes per integration step
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    x0 = HenonHeiles().sample_initial_x()
    print(x0)
    # print(DoublePendulum().calc_E(x0))
    # x0 = np.array([0, (1 / 1) ** 0.5, 0, 0])

    d = x0.shape[0]  # dimension of the ODE state space
    error_tol = 0.0001

    scaler = load(open("test_scaler.pkl", "rb"))

    env = ODEEnv(fun=HenonHeiles(), max_iterations=10000, initial_step_size=step_sizes[0],
                 step_size_range=(step_sizes[0], step_sizes[-1]), reward_fun=None,
                 error_tol=error_tol, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=20)

    predictor = PredictorQODE(step_sizes=step_sizes,
                              model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
                                                         filename=None, lr=0.01, memory=memory),
                              scaler=scaler)

    best_predictors = BestPredictors().load()
    print(best_predictors.errors)
    print(best_predictors.rewards)
    # print(best_predictors.nfev_training)
    predictor.model.set_weights(best_predictors.best_by_reward())
    # predictor.model.set_weights(best_predictors.weights[1])

    # predictor = PredictorConstODE(0.015)  # 0.56
    # integrator = ClassicRungeKutta()
    integrator = RKDP()
    # integrator.b = best_predictors.best_integrator_by_reward()
    # print(integrator.b)
    # integrator = load("integrator_optim_weights.pkl")

    t1 = 200
    reward, num_evals = integrate_env(predictor, integrator, env, t1=t1, plot=False)

    fig, ax = plot_trajectory(
        x0=env.x0,
        timesteps=env.timesteps,
        nodes=env.nodes,
        errors=env.errors,
        deltas=env.deltas,
        error_tolerance=env.error_tol,
    )

    print(env.nodes)
    plt.savefig("traj.pdf")
    plt.show()

    print("reward: {}".format(reward))
    print("nfev: {}".format(num_evals / t1))
    print("mean error: {}".format(np.mean(env.errors)))
    print("min error, max error: {}, {}".format(*np.round((np.min(env.errors), np.max(env.errors)), 5)))
    print("min stepsize: {}".format(np.min(env.deltas)))
    print("max stepsize: {}".format(np.max(env.deltas)))


# def one_fun_meta():
#     """
#     Integrate and plot one ODE (using MetaQODE) and print important statistics.
#     """
#     d = 2  # dimension of the ODE state space
#     x0 = np.array([1, 1])
#     basis_learners = []
#     t1 = 50
#
#     # define basis learner
#     step_sizes = [0.25, 0.27, 0.29, 0.31, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48]
#     dim_state = 6  # nodes per integration step
#     dim_action = len(step_sizes)
#     memory = 0  # how many integration steps the predictor can look back
#
#     scaler = StandardScaler()
#     scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
#     scaler.mean_[0] = 0.33
#     scaler.scale_ = np.ones((dim_state * d + 1) * (memory + 1))
#     scaler.scale_[0] = 0.1
#
#     basis_learners.append(PredictorQODE(step_sizes=step_sizes,
#                                         model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
#                                                                    filename='predictorODE', memory=memory),
#                                         scaler=scaler))
#     basis_learners.append(PredictorConstODE(0.1))
#     basis_learners.append(PredictorConstODE(0.05))
#     basis_learners.append(PredictorConstODE(0.01))
#     basis_learners.append(PredictorConstODE(0.005))
#     basis_learners.append(PredictorConstODE(0.001))
#
#     memory = 1
#     scaler = StandardScaler()
#     scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
#     scaler.mean_[0] = 0.33
#     scaler.scale_ = np.ones((dim_state * d + 1) * (memory + 1))
#     scaler.scale_[0] = 0.1
#
#     # define meta learner
#     metalearner = MetaQODE(basis_learners,
#                            model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=len(basis_learners),
#                                                       filename='metaODE', memory=memory, lr=0.001),
#                            scaler=scaler, use_idx=False)
#
#     # define environment
#     env = ODEEnv(fun=Pendulum(switchpoints=(0.05, 3.3)), max_iterations=10000, initial_step_size=0.25,
#                  step_size_range=(0.005, 0.48),
#                  error_tol=0.00001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=100,
#                  stepsize_to_idx=stepsize_to_idx)
#     integrator = RKDP()
#
#     reward, num_evals = integrate_env(metalearner, integrator, env, t1=t1, plot=True)
#     print('avg. error: {}'.format(np.mean(env.errors)))
#     print('evals: {}'.format(env.evals / t1))
#     print(env.fun.switch_times)


def pareto_model():
    """
    Find performance (avg. error, avg. evals) of PredictorODE w.r.t. a function class.
    """
    num_samples = 20

    step_sizes = [0.02, 0.022, 0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070]
    dim_state = 6  # nodes per integration step
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    x0 = np.array([0, 0, 0])
    d = len(x0)  # dimension of the ODE state space

    # scaler = StandardScaler()
    scaler = load(open("test_scaler.pkl", "rb"))

    t1 = 100

    env = ODEEnv(fun=LorenzSystem(), max_iterations=10000, initial_step_size=step_sizes[0],
                 error_tol=0.0001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=t1)

    predictor = PredictorQODE(step_sizes=step_sizes,
                              model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
                                                         filename=None, lr=0.01, memory=memory),
                              scaler=scaler)

    best_predictors = BestPredictors().load()
    print(best_predictors.rewards)
    print(best_predictors.errors)
    predictor.model.set_weights(best_predictors.best_by_reward())
    predictor.model.set_weights(best_predictors.weights[3])

    integrator = RKDP()
    print(integrator.b)
    integrator.b = best_predictors.best_integrator_by_reward()
    print(integrator.b)
    integrator.b = best_predictors.weights_integrator[3]

    performance = np.zeros((num_samples, 2))
    for i in range(num_samples):
        if i % 10 == 0:
            print(i)

        # env.x0 = DoublePendulum().sample_initial_x(20)
        env.x0 = 20 * np.random.random(3) - 10
        env.x0[2] += 25

        env.reset(integrator)
        integrate_env(predictor, integrator, env, t1=t1)

        performance[i, :] = [np.mean(env.errors), env.evals / t1]

    print(np.mean(performance, axis=0))
    np.save('pareto_model_optim_weights.npy', performance)

    # plt.hist(errors, 25)
    # plt.show()


def pareto_const_predictor():
    """
    Find performance (avg. error, avg. evals) of constant step size choice w.r.t. a function class.
    """
    num_samples = 1000

    step_sizes = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
    dim_state = 6  # nodes per integration step
    d = 2  # dimension of the ODE state space
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    x0 = np.array([1.0, 1.0])

    scaler = StandardScaler()
    # scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
    # scaler.scale_ = 100 * np.ones((dim_state * d + 1) * (memory + 1))
    # scaler.scale_[0] = 0.1

    scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
    scaler.scale_ = np.ones((dim_state * d + 1) * (memory + 1))

    env = ODEEnv(fun=LorenzSystem(), max_iterations=10000, initial_step_size=step_sizes[0],
                 error_tol=0.0001, nodes_per_integ=dim_state, memory=memory, x0=x0)
    integrator = RKDP()

    paretos = []
    for action in range(len(step_sizes)):
        print("action: {}".format(action))
        predictor = PredictorConstODE(action)
        errors = []
        steps = []
        t1s = []

        for i in range(num_samples):
            env.reset(integrator)
            integrate_env(predictor, integrator, env)
            errors.append(np.mean(env.errors))
            steps.append(env.evals)
            t1s.append(env.timesteps[-1])

        print(np.mean(steps))
        print(np.mean(errors))
        print("")
        paretos.append((np.mean(errors), np.mean(steps)))

    paretos = np.array(paretos)
    np.save("pareto_const.npy", paretos)


def pareto_ode45():
    """
    Find performance (avg. error, avg. evals) of ode45 (RK45) w.r.t. a function class.
    """
    num_samples = 20
    # f = LorenzSystem()
    # f = Pendulum(switchpoints=(0.05, 3.3))
    f = HenonHeiles()
    x0 = np.array([0, (1 / 6) ** 0.5, 0, (1 / 6) ** 0.5])
    t1 = 500
    tols = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    tols = [3.5e-6, 4e-6, 4.8e-6]
    tols = [6.5e-5, 8e-5, 1e-4]
    # tols = [7.8e-5]

    paretos = []
    paretos_norep = []
    for tol in tols:
        print("tol: {}".format(tol))
        errors = []
        fevals = []
        fevals_norep = []
        for i in range(num_samples):
            f.reset()
            x0 = f.sample_initial_x()
            sol = solve_ivp(f, (0, t1), x0, atol=tol, rtol=tol)
            x_predict = sol.y
            t_predict = sol.t
            # plt.plot(t_predict, x_predict[0, :])
            # plt.show()

            this_errors = []
            for idx in range(t_predict.shape[0] - 1):
                x_true = f.solve(t_predict[idx], x_predict[:, idx], t_predict[idx + 1])
                this_errors.append(np.linalg.norm(x_predict[:, idx + 1] - x_true))
                # print(t_predict[idx + 1] - t_predict[idx], np.linalg.norm(x_predict[:, idx + 1] - x_true), x_predict[:, idx + 1] - x_true)
                # print('')

            errors.append(np.mean(this_errors))
            # print((sol.t.shape[0] - 1) * 6 + 2, sol.nfev)
            fevals.append(sol.nfev / t1)
            fevals_norep.append(((sol.t.shape[0] - 1) * 6 + 1) / t1)

        paretos.append((np.mean(errors), np.mean(fevals)))
        paretos_norep.append((np.mean(errors), np.mean(fevals_norep)))

        print("error: {}".format(np.mean(errors)))
        print("fevals: {}".format(np.mean(fevals)))
        print('fevals norej: {}'.format(np.mean(fevals_norep)))
        print('')

    paretos = np.array(paretos)
    np.save('pareto_ode45.npy', paretos)
    np.save('pareto_ode45_norepcount.npy', paretos_norep)


def save_pareto_plot():
    """
    Plot performances (avg. error, avg. evals).
    """
    pareto_mod = np.load("pareto_model.npy")
    pareto_mod_optim = np.load("pareto_model_optim_weights.npy")
    pareto_ode = np.load("pareto_ode45.npy")[:-1, :]
    pareto_ode_norepcount = np.load("pareto_ode45_norepcount.npy")[:-1, :]

    fig, ax = plot_pareto(
        model_data=pareto_mod,
        ode_data=pareto_ode,
        ode_norep_data=pareto_ode_norepcount,
        opt_model_data=pareto_mod_optim,
        yrange=(130, 201),
        # xrange=(3e-5, 8e-5)
    )

    fig.savefig("pareto.pdf")
    plt.show()


if __name__ == "__main__":
    # onefun()
    # one_fun_meta()
    # pareto_model()
    # pareto_const_predictor()
    # pareto_ode45()
    save_pareto_plot()
