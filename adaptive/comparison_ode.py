import numpy as np
import math
from matplotlib import pyplot as plt
from adaptive.environments import ODEEnv
from adaptive.integrator import IntegratorODE, ClassicRungeKutta, RKDP
from functions import Rotation, LorenzSystem, Pendulum
from adaptive.experience import ExperienceODE
from adaptive.predictor import PredictorODE, PredictorQODE, PredictorConstODE, MetaQODE
from adaptive.build_models import build_value_model, build_value_modelODE
from adaptive.performance_tracker import PerformanceTrackerODE
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

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
            raise ValueError('Give value for t1!')

    state = env.reset(reset_params=False, integrator=integrator)
    reward = 0
    while True:
        step_size = predictor(state)
        next_state, r, _, info = env.iterate(step_size, integrator)
        reward += r
        state = next_state.copy()
        if env.timesteps[-1] >= t1:
            break

    num_evals = env.evals
    # for idx, node in enumerate(env.nodes[-env.nodes_per_integ + 1:-1]):
    #     if node > x1:
    #         num_evals += idx - (env.nodes_per_integ - 2)  # delete not needed evaluations
    #         break

    if plot:
        env.plot()

    return reward, num_evals


def onefun():
    """
    Integrate and plot one ODE (using PredictorODE) and print important statistics.
    """
    # step_sizes = [0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070]
    step_sizes = [0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070]
    dim_state = 6  # nodes per integration step
    d = 3  # dimension of the ODE state space
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    x0 = np.array([10.0, 10.0, 10.0])
    # x0 = np.random.rand(3) * 20 - 10
    # print(x0)

    scaler = StandardScaler()
    scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
    scaler.mean_[0] = -0.045
    scaler.scale_ = 10 * np.ones((dim_state * d + 1) * (memory + 1))
    scaler.scale_[0] = 0.1

    # scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
    # scaler.scale_ = np.ones((dim_state * d + 1) * (memory + 1))

    env = ODEEnv(fun=LorenzSystem(), max_iterations=10000, initial_step_size=0.025,
                 step_size_range=(step_sizes[0], step_sizes[-1]),
                 error_tol=0.0001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=100)

    predictor = PredictorQODE(step_sizes=step_sizes,
                              model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
                                                         filename='predictorODE', lr=0.01, memory=memory),
                              scaler=scaler)

    # predictor = PredictorConstODE(0.05)
    # integrator = ClassicRungeKutta()
    integrator = RKDP()

    reward, num_evals = integrate_env(predictor, integrator, env, t1=2, plot=True)

    print("reward: {}".format(reward))
    print("nfev: {}".format(num_evals))
    print("mean error: {}".format(np.mean(env.errors)))
    print("min error, max error: {}, {}".format(*np.round((np.min(env.errors), np.max(env.errors)), 5)))
    print("min stepsize: {}".format(np.min(env.deltas)))
    print("max stepsize: {}".format(np.max(env.deltas)))

    # le = len(env.fun.switch_times)
    # for t, x in zip(env.fun.switch_times[:le], env.fun.state_at_switch_times[:le]):
    #     print(t, x, np.linalg.norm(env.fun(t * 0.9999999999999, x)))


def one_fun_meta():
    """
    Integrate and plot one ODE (using MetaQODE) and print important statistics.
    """
    d = 2  # dimension of the ODE state space
    x0 = np.array([1, 1])
    basis_learners = []
    t1 = 50

    # define basis learner
    step_sizes = [0.25, 0.27, 0.29, 0.31, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48]
    dim_state = 6  # nodes per integration step
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back

    scaler = StandardScaler()
    scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
    scaler.mean_[0] = 0.33
    scaler.scale_ = np.ones((dim_state * d + 1) * (memory + 1))
    scaler.scale_[0] = 0.1

    basis_learners.append(PredictorQODE(step_sizes=step_sizes,
                                        model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
                                                                   filename='predictorODE', memory=memory),
                                        scaler=scaler))
    basis_learners.append(PredictorConstODE(0.1))
    basis_learners.append(PredictorConstODE(0.05))
    basis_learners.append(PredictorConstODE(0.01))
    basis_learners.append(PredictorConstODE(0.005))
    basis_learners.append(PredictorConstODE(0.001))

    memory = 1
    scaler = StandardScaler()
    scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
    scaler.mean_[0] = 0.33
    scaler.scale_ = np.ones((dim_state * d + 1) * (memory + 1))
    scaler.scale_[0] = 0.1

    # define meta learner
    metalearner = MetaQODE(basis_learners,
                           model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=len(basis_learners),
                                                      filename='metaODE', memory=memory, lr=0.001),
                           scaler=scaler, use_idx=False)

    # define environment
    env = ODEEnv(fun=Pendulum(switchpoints=(0.05, 3.3)), max_iterations=10000, initial_step_size=0.25,
                 step_size_range=(0.005, 0.48),
                 error_tol=0.00001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=100,
                 stepsize_to_idx=stepsize_to_idx)
    integrator = RKDP()

    reward, num_evals = integrate_env(metalearner, integrator, env, t1=t1, plot=True)
    print('avg. error: {}'.format(np.mean(env.errors)))
    print('evals: {}'.format(env.evals / t1))
    print(env.fun.switch_times)


def pareto_model():
    """
    Find performance (avg. error, avg. evals) of PredictorODE w.r.t. a function class.
    """
    num_samples = 1

    step_sizes = [0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070]
    dim_state = 6  # nodes per integration step
    d = 3  # dimension of the ODE state space
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    x0 = np.array([10, 10, 10])

    scaler = StandardScaler()
    scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))
    scaler.mean_[0] = -0.045
    scaler.scale_ = 10 * np.ones((dim_state * d + 1) * (memory + 1))
    scaler.scale_[0] = 0.1

    t1 = 100

    env = ODEEnv(fun=LorenzSystem(), max_iterations=10000, initial_step_size=0.025,
                 error_tol=0.0001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=t1)

    predictor = PredictorQODE(step_sizes=step_sizes,
                              model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
                                                         filename='predictorODE', lr=0.01, memory=memory),
                              scaler=scaler)

    # predictor = PredictorConstODE(5)
    # integrator = ClassicRungeKutta()
    integrator = RKDP()

    errors = []
    steps = []
    t1s = []
    for i in range(num_samples):
        if i % 10 == 0:
            print(i)

        # env.x0 = np.random.rand(3) * 20 - 10
        env.reset(integrator)
        integrate_env(predictor, integrator, env, t1=t1)
        errors.append(np.mean(env.errors))
        steps.append(env.evals / t1)
        t1s.append(env.timesteps[-1])

    print(np.mean(steps))
    print(np.mean(errors))
    print(np.mean(t1s))
    print(np.quantile(errors, 0.9))
    np.save('pareto_model.npy', np.array([np.mean(errors), np.mean(steps)]))

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

    env = ODEEnv(fun=Rotation(), max_iterations=10000, initial_step_size=0.6, step_size_range=step_sizes,
                 error_tol=0.0001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=20)
    integrator = RKDP()

    paretos = []
    for action in range(len(step_sizes)):
        print('action: {}'.format(action))
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
        print('')
        paretos.append((np.mean(errors), np.mean(steps)))

    paretos = np.array(paretos)
    np.save('pareto_const.npy', paretos)


def pareto_ode45():
    """
    Find performance (avg. error, avg. evals) of ode45 (RK45) w.r.t. a function class.
    """
    num_samples = 1
    # f = LorenzSystem()
    f = Pendulum(switchpoints=(0.05, 3.3))
    x0 = np.array([1, 1])
    t1 = 100
    # tols = [1e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 5e-4]
    tols = [7.8e-5]

    paretos = []
    paretos_norep = []
    for tol in tols:
        print("tol: {}".format(tol))
        errors = []
        fevals = []
        fevals_norep = []
        for i in range(num_samples):
            f.reset()
            sol = solve_ivp(f, (0, t1), x0, atol=tol, rtol=tol)
            x_predict = sol.y
            t_predict = sol.t

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


def plot_pareto():
    """
    Plot performances (avg. error, avg. evals).
    """
    # 0.0008868960659465355, 19.5
    # pareto_const = np.load('pareto_const.npy')
    pareto_mod = np.load('pareto_model.npy')
    # pareto_mod = [1.3763545225205223e-05, 23.4]
    pareto_ode = np.load('pareto_ode45.npy')
    pareto_ode_norepcount = np.load('pareto_ode45_norepcount.npy')

    fig = plt.figure(figsize=(5, 4), dpi=300)
    plt.xlim((2e-5, 3e-4))
    plt.ylim((115, 220))

    # plt.loglog(pareto_const[:, 0], pareto_const[:, 1], 'bx-', label='const')
    plt.loglog(pareto_mod[0], pareto_mod[1], 'r^', label='model')
    plt.loglog(pareto_ode[:, 0], pareto_ode[:, 1], 'gx-', label='RK45')
    plt.loglog(pareto_ode_norepcount[:, 0], pareto_ode_norepcount[:, 1], 'bx-', label='RK45 (rejections not counted)')

    plt.legend()
    plt.xlabel('error per RK step')
    plt.ylabel('number of function eval.')
    plt.grid(which='both')
    plt.tight_layout()
    plt.savefig('pareto.png')
    plt.show()


if __name__ == '__main__':
    # onefun()
    # one_fun_meta()
    # pareto_model()
    # pareto_const_predictor()
    pareto_ode45()
    # plot_pareto()
