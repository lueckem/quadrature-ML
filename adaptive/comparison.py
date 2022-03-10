import numpy as np
from adaptive.environments import IntegrationEnv
from adaptive.build_models import build_value_model
from functions import Sinus, Pulse, StaggeredSinus, SuperposeSinus
from joblib import dump, load
from adaptive.adapt_simps import AdaptSimps, AdaptSimpsConstEvals, Simps
from adaptive.predictor import *
from adaptive.integrator import Integrator, Simpson
from adaptive.error_estimator import Estimator


def integrate_env(predictor, integrator, env, x0, x1, plot=False, estimator=None):
    """
    Integrates the environment from x0 to x1 using the specified predictor and integrator.

    Parameters
    ----------
    predictor : Predictor
    integrator : Integrator
    env : IntegrationEnv
        does not change the function
    x0 : float
    x1 : float
    plot : bool, optional
    estimator : Estimator, optional

    Returns
    -------
    integ : float
        integral from x0 to last_node
    num_evals : int
        number of function evaluations
    last_node : float
        x1 is not hit exactly depending on chosen step_sizes
    errors : list[float]
        stepwise absolute errors
    """
    env.x0 = x0
    state = env.reset(reset_params=False, integrator=integrator)
    # states = [state[0].copy()]

    while True:
        action = predictor(state)
        next_state, _, _, info = env.iterate(action, integrator, estimator=estimator)
        state = next_state
        # states.append(state[0].copy())
        if env.nodes[-1] >= x1:
            break

    num_evals = env.evals
    # for idx, node in enumerate(env.nodes[-env.nodes_per_integ + 1:-1]):
    #     if node > x1:
    #         num_evals += idx - (env.nodes_per_integ - 2)  # delete not needed evaluations
    #         break

    if plot:
        env.plot(x_min=x0, x_max=x1, episode=0, save=True)

    # scaler = StandardScaler()
    # scaler.fit(states)
    # print(scaler.scale_)
    # print(scaler.mean_)
    # dump(scaler, "scaler_integ.pkl")

    return env.integral, num_evals, env.nodes[-1], env.errors


def one_fun():
    x0 = 0.0
    x1 = 10.0
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    dim_state = 3
    dim_action = len(step_sizes)
    memory = 0
    scaler = load('model_quad/model_sinus/Simpson/scaler.bin')

    env = IntegrationEnv(fun=Sinus(), max_iterations=10000, initial_step_size=0.075,
                         error_tol=7.5e-6, nodes_per_integ=dim_state, memory=memory,
                         x0=0, max_dist=10, step_size_range=(step_sizes[0], step_sizes[-1]))
    predictor = PredictorQ(step_sizes,
                           build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
                           scaler=scaler)
    integrator = Simpson()
    integ, evals, x1, _ = integrate_env(predictor, integrator, env, x0, x1, plot=True)
    print('new x1: {}'.format(x1))
    print('Predictor error: {}'.format(abs(env.fun.integral(x0, x1) - integ)))
    print('Predictor evals: {}'.format(evals))

    env.reset(integrator, reset_params=False)
    asr = AdaptSimpsConstEvals(env.fun, x0, x1)
    integ = asr(evals)
    print('ASR error: {}'.format(abs(env.fun.integral(x0, x1) - integ)))
    print('ASR evals: {}'.format(asr.evals))
    asr.plot()

    env.reset(integrator, reset_params=False)
    simps = Simps(env.fun, x0, x1)
    integ_simps = simps(num_evals=evals)
    print('Simpson error: {}'.format(abs(env.fun.integral(x0, x1) - integ_simps)))
    print('Simpson evals: {}'.format(simps.evals))
    simps.plot()


def compare():
    """ compare predictor to adaptive simpson rule based on an average performance on a sample of functions """
    x0 = 0.0
    num_samples = 200
    error_predictor = 0.0
    error_simpson = 0.0

    step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    dim_state = 3
    dim_action = len(step_sizes)
    env = IntegrationEnv(fun=SuperposeSinus(5), max_iterations=256, initial_step_size=0.2,
                         step_sizes=step_sizes, error_tol=0.0005)
    predictor = PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
                           load('scaler.bin'))

    for i in range(num_samples):
        if i % 10 == 0:
            print(i)

        x1 = 200.0
        env.reset()
        integ_pred, evals, x1, _ = integrate_env(predictor, Simpson(), env, x0, x1)
        integ = env.fun.integral(x0, x1)
        env.reset(reset_params=False)

        # asr = AdaptSimpsConstEvals(env.fun, x0, x1)
        # integ_simps = asr(evals)
        asr = Simps(env.fun, x0, x1)
        integ_simps = asr(num_evals=evals)

        error_predictor += abs(integ - integ_pred)
        error_simpson += abs(integ - integ_simps)

    error_simpson /= num_samples
    error_predictor /= num_samples

    print('Predictor error: {}'.format(error_predictor))
    print('ASR error: {}'.format(error_simpson))


if __name__ == '__main__':
    # compare()
    one_fun()
