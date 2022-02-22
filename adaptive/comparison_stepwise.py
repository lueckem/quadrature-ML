import numpy as np
from adaptive.environments import IntegrationEnv
from adaptive.build_models import build_value_model, build_estimator_model
from functions import Sinus, Pulse, StaggeredSinus, SuperposeSinus, Polynomial, BrokenPolynomial, DoublePendulumInteg
from joblib import dump, load
from scipy.integrate import quad, romberg, quad_vec
from adaptive.adapt_simps import AdaptSimps, AdaptSimpsConstEvals, Simps
from adaptive.boole import BoolesRule
from adaptive.romberg import Romberg
from adaptive.predictor import *
from adaptive.comparison import integrate_env
from adaptive.integrator import Simpson, IntegratorLinReg, Boole, Kronrod21, Gauss21
from matplotlib import pyplot as plt
from adaptive.plots import plot_pareto


def one_fun():
    x0 = 0
    x1 = 50
    step_sizes = np.geomspace(0.1, 0.7, 20)
    dim_state = 22
    dim_action = len(step_sizes)
    memory = 0
    scaler = load("scaler_integ.pkl")

    integrator = Gauss21()
    # integrator = IntegratorLinReg(step_sizes, load("linreg_integrator.bin"), integrator)
    env = IntegrationEnv(fun=DoublePendulumInteg(x0, x1), max_iterations=10000, initial_step_size=step_sizes[0],
                         error_tol=1e-7, nodes_per_integ=integrator.num_nodes, memory=memory,
                         x0=x0, max_dist=x1 - x0, step_size_range=(step_sizes[0], step_sizes[-1]))
    predictor = PredictorQ(step_sizes,
                           build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
                           scaler=scaler)
    # predictor = PredictorConst(step_sizes[0])
    # predictor = PredictorRandom(step_sizes)
    # integrator = IntegratorLinReg(step_sizes, load('linreg_models.bin'), load('scaler.bin'))

    _, evals, x1, errors = integrate_env(predictor, integrator, env, x0, x1, plot=True)
    num_subintervals = len(env.nodes) - 1
    print('new x1: {}'.format(x1))
    print('Predictor error total: {}'.format(np.sum(errors)))
    print('Predictor error per step: {}'.format(np.mean(errors)))
    print('Predictor evals: {}'.format(evals))
    print('')

    tol = 1e-3
    out = quad(env.fun, x0, x1, epsabs=tol, epsrel=tol, full_output=True, limit=int(x1 - x0) * 50)
    print(f"Quad error total: {abs(out[0] - env.fun.integral(x0, x1))}")
    print(f"Quad evals {out[2]['neval']}")
    print()

    # env.reset(integrator, reset_params=False)
    # asr = AdaptSimpsConstEvals(env.fun, x0, x1)
    # asr(200)
    # errors = asr.stepwise_errors
    # print('ASR error total: {}'.format(np.sum(errors)))
    # print('ASR error per step: {}'.format(np.mean(errors)))
    # print('ASR evals: {}'.format(asr.evals))
    # print('')
    # asr.plot()
    #
    env.reset(integrator, reset_params=False)
    simps = Simps(env.fun, x0, x1)
    integ_simps, errors = simps(num_evals=evals, stepwise_error=True)
    # integ_simps, errors = simps(step_size=0.11, stepwise_error=True)
    print('Simpson error total: {}'.format(np.sum(errors)))
    print('Simpson error per step: {}'.format(np.mean(errors)))
    print('Simpson evals: {}'.format(simps.evals))
    print('')
    # simps.plot()
    #
    # env.reset(integrator, reset_params=False)
    # rom = Romberg(env.fun, x0, x1, tol=0.0005, order=2)
    # integ, errors = rom(0.15, stepwise_errors=True)
    # print('Romberg error total: {}'.format(np.sum(errors)))
    # print('Romberg error per step: {}'.format(np.mean(errors)))
    # print('Romberg evals: {}'.format(rom.evals))
    # rom.plot()


# def one_fun_boole():
#     x0 = 0
#     x1 = 10
#     step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]
#     # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
#     dim_state = 5
#     dim_action = len(step_sizes)
#     memory = 1
#     env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.1,
#                          step_sizes=step_sizes, error_tol=0.000001, memory=memory, nodes_per_integ=dim_state)
#     predictor = PredictorQ(
#         build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor', memory=memory),
#         load('scaler_boole_mem1.bin'))
#     # integrator = IntegratorLinReg(step_sizes, load('linreg_models.bin'), load('scaler.bin'))
#     integrator = Boole()
#
#     _, evals, x1, errors = integrate_env(predictor, integrator, env, x0, x1, plot=True)
#     print('new x1: {}'.format(x1))
#     print('Predictor error total: {}'.format(np.sum(errors)))
#     print('Predictor error per step: {}'.format(np.mean(errors)))
#     print('Predictor evals: {}'.format(evals))
#     print('')
#
#     env.reset(reset_params=False)
#     booles = BoolesRule(env.fun, x0, x1)
#     integ_simps, errors = booles(num_evals=evals, stepwise_error=True)
#     print('Boole error total: {}'.format(np.sum(errors)))
#     print('Boole error per step: {}'.format(np.mean(errors)))
#     print('Boole evals: {}'.format(booles.evals))
#     print('')
#     booles.plot()
#
#     env.reset(reset_params=False)
#     simps = Simps(env.fun, x0, x1)
#     integ_simps, errors = simps(num_evals=evals, stepwise_error=True)
#     print('Simpson error total: {}'.format(np.sum(errors)))
#     print('Simpson error per step: {}'.format(np.mean(errors)))
#     print('Simpson evals: {}'.format(simps.evals))
#     print('')
#     simps.plot()
#
#     env.reset(reset_params=False)
#     rom = Romberg(env.fun, x0, x1, tol=0.0005, order=3)
#     integ, errors = rom(0.15, stepwise_errors=True)
#     print('Romberg error total: {}'.format(np.sum(errors)))
#     print('Romberg error per step: {}'.format(np.mean(errors)))
#     print('Romberg evals: {}'.format(rom.evals))
#     rom.plot()


# def compare_simps_tol():
#     """ compare model to simpson rule, stepsize for simps is adapted to match tol. """
#     x0 = 0.0
#     num_samples = 2000
#     error_predictor = []
#     error_simps = []
#     evals_simps = []
#     evals_predictor = []
#
#     step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
#     # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
#     dim_state = 3
#     dim_action = len(step_sizes)
#     env = IntegrationEnv(fun=SuperposeSinus(5), max_iterations=256, initial_step_size=0.15,
#                          step_sizes=step_sizes, error_tol=0.0005)
#     predictor = PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
#                            load('scaler.bin'))
#
#     for i in range(num_samples):
#         if i % 10 == 0:
#             print(i)
#
#         x1 = 20.0
#
#         # model
#         env.reset()
#         _, evals, x1, errors = integrate_env(predictor, Simpson(), env, x0, x1)
#         error_pred_step = np.mean(errors)
#         env.reset(reset_params=False)
#
#         # simpson
#         step_size = 0.177
#         simp = Simps(env.fun, x0, x1)
#         _, error_simps_step = simp(step_size=step_size, stepwise_error=True)
#         error_simps_step = np.mean(error_simps_step)
#         evals_simps_step = simp.evals
#         # for j in range(10):
#         #     simp = Simps(env.fun, x0, x1)
#         #     integ_rom, error_simps_step = simp(step_size=step_size, stepwise_error=True)
#         #     error_simps_step /= (simp.evals - 1.0) / 2.0
#         #     evals_simps_step = simp.evals
#         #     if error_simps_step < 0.0001:
#         #         step_size *= 1.5
#         #     elif error_simps_step > 0.0005:
#         #         step_size /= 2.0
#         #     else:
#         #         break
#
#         error_predictor.append(error_pred_step)
#         error_simps.append(error_simps_step)
#         evals_predictor.append(evals)
#         evals_simps.append(evals_simps_step)
#
#     error_simps = np.array(error_simps)
#     not_converged = np.concatenate((error_simps[error_simps > 0.0005], error_simps[error_simps < 0.0001]))
#     if len(not_converged > 0):
#         print('simps did not converge in some cases:')
#         print(not_converged)
#
#     mean_error_predictor = np.mean(error_predictor)
#     var_error_predictor = np.var(error_predictor)
#
#     mean_error_simps = np.mean(error_simps)
#     var_error_simps = np.var(error_simps)
#
#     mean_evals_predictor = np.mean(evals_predictor)
#     var_evals_predictor = np.var(evals_predictor)
#
#     mean_evals_simps = np.mean(evals_simps)
#     var_evals_simps = np.var(evals_simps)
#
#     print('Avg. predictor number of function evaluations per episode: {}'.format(mean_evals_predictor))
#     print('Avg. predictor error per step: {}'.format(mean_error_predictor))
#     print('Avg. simpson number of function evaluations per episode: {}'.format(mean_evals_simps))
#     print('Avg. simpson error per step: {}'.format(mean_error_simps))
#     print('')
#     print('Variance of predictor number of function evaluations per episode: {}'.format(var_evals_predictor))
#     print('Variance of predictor error per step: {}'.format(var_error_predictor))
#     print('Variance of simpson number of function evaluations per episode: {}'.format(var_evals_simps))
#     print('Variance of simpson error per step: {}'.format(var_error_simps))


# def compare_romberg():
#     x0 = 0.0
#     num_samples = 100
#     error_predictor = []
#     error_rom = []
#     evals_rom = []
#     evals_predictor = []
#
#     step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
#     # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
#     dim_state = 3
#     dim_action = len(step_sizes)
#     env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.15,
#                          step_sizes=step_sizes, error_tol=0.0005)
#     predictor = PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
#                            load('scaler.bin'))
#
#     for i in range(num_samples):
#         if i % 10 == 0:
#             print(i)
#
#         x1 = 20.0
#
#         # model
#         env.reset()
#         _, evals, x1, errors = integrate_env(predictor, Simpson(), env, x0, x1)
#         error_pred_step = np.mean(errors)
#         env.reset(reset_params=False)
#
#         # romberg
#         tol = 0.0003
#
#         rom = Romberg(env.fun, x0, x1, tol=tol, order=2)
#         integ_rom, errors = rom(0.15, True)
#         error_rom_step = np.mean(errors)
#         evals_rom_step = rom.evals
#
#         # for j in range(10):
#         #     rom = Romberg(env.fun, x0, x1, tol=tol, order=2)
#         #     integ_rom, errors = rom(0.15, True)
#         #     error_rom_step = np.mean(errors)
#         #     evals_rom_step = rom.evals
#         #     if error_rom_step < 0.0001:
#         #         tol *= 2.0
#         #     elif error_rom_step > 0.0005:
#         #         tol /= 3.0
#         #     else:
#         #         break
#
#         error_predictor.append(error_pred_step)
#         error_rom.append(error_rom_step)
#         evals_predictor.append(evals)
#         evals_rom.append(evals_rom_step)
#
#     # error_rom = np.array(error_rom)
#     # not_converged = np.concatenate((error_rom[error_rom > 0.0005], error_rom[error_rom < 0.0001]))
#     # if len(not_converged > 0):
#     #     print('romberg did not converge in some cases:')
#     #     print(not_converged)
#
#     mean_error_predictor = np.mean(error_predictor)
#     var_error_predictor = np.var(error_predictor)
#
#     mean_error_rom = np.mean(error_rom)
#     var_error_rom = np.var(error_rom)
#
#     mean_evals_predictor = np.mean(evals_predictor)
#     var_evals_predictor = np.var(evals_predictor)
#
#     mean_evals_rom = np.mean(evals_rom)
#     var_evals_rom = np.var(evals_rom)
#
#     print('Avg. predictor number of function evaluations per episode: {}'.format(mean_evals_predictor))
#     print('Avg. predictor error per step: {}'.format(mean_error_predictor))
#     print('Avg. romberg number of function evaluations per episode: {}'.format(mean_evals_rom))
#     print('Avg. romberg error per step: {}'.format(mean_error_rom))
#     print('')
#     print('Variance of predictor number of function evaluations per episode: {}'.format(var_evals_predictor))
#     print('Variance of predictor error per step: {}'.format(var_error_predictor))
#     print('Variance of rom number of function evaluations per episode: {}'.format(var_evals_rom))
#     print('Variance of rom error per step: {}'.format(var_error_rom))


# def pareto_simpson():
#     """ Find pairs (mean error, avg steps) for equidistand simpson rules with different step sizes."""
#     x0 = -1
#     x1 = 1
#     num_samples = 5000
#     step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
#
#     env = IntegrationEnv(fun=BrokenPolynomial(), max_iterations=1000, initial_step_size=0.1,
#                          step_sizes=step_sizes, error_tol=0.0005)
#     step_sizes = np.linspace(0.05, 0.4, 16)
#     paretos = []
#
#     for step_size in step_sizes:
#         print(step_size)
#         this_errors = []
#         for _ in range(num_samples):
#             env.reset()
#             simp = Simps(env.fun, x0, x1)
#             _, error_simps_step = simp(step_size=step_size, stepwise_error=True)
#             this_errors.append(np.mean(error_simps_step))
#         paretos.append((np.mean(this_errors), (x1 - x0) / step_size + 1))
#
#         # plt.hist(this_errors, 25)
#         # plt.show()
#
#     paretos = np.array(paretos)
#     np.save('pareto_simpson', paretos)
#
#
# def pareto_boole():
#     """ Find pairs (mean error, avg steps) for equidistand booles rules with different step sizes."""
#     x0 = 0.0
#     x1 = 20.0
#     num_samples = 4000
#     step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]
#
#     env = IntegrationEnv(fun=SuperposeSinus(5), max_iterations=256, initial_step_size=0.05,
#                          step_sizes=step_sizes, error_tol=0.0005, memory=0)
#     step_sizes = np.linspace(0.025, 0.25, 31)
#     paretos = []
#
#     for step_size in step_sizes:
#         print(step_size)
#         this_errors = []
#         for _ in range(num_samples):
#             env.reset()
#             simp = BoolesRule(env.fun, x0, x1)
#             _, error_simps_step = simp(step_size=step_size, stepwise_error=True)
#             this_errors.append(np.mean(error_simps_step))
#         paretos.append((np.mean(this_errors), (x1 - x0) / step_size + 1))
#
#     paretos = np.array(paretos)
#     np.save('pareto_boole', paretos)
#
#
# def pareto_asr():
#     x0 = 0
#     x1 = 20
#     num_samples = 5000
#     step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
#
#     env = IntegrationEnv(fun=SuperposeSinus(5), max_iterations=1000, initial_step_size=0.1,
#                          step_sizes=step_sizes, error_tol=0.0005)
#     nevals = [75, 90, 108, 130, 155, 186, 223, 268, 322, 386]
#     paretos = []
#
#     for neval in nevals:
#         print(neval)
#         this_errors = []
#         this_evals = []
#         for _ in range(num_samples):
#             env.reset()
#             asr = AdaptSimpsConstEvals(env.fun, x0, x1)
#             asr(neval)
#             stepwise_errors = asr.stepwise_errors
#             this_errors.append(np.mean(stepwise_errors))
#             this_evals.append(asr.evals)
#         paretos.append((np.mean(this_errors), np.mean(this_evals)))
#
#         # plt.hist(this_errors, 25)
#         # plt.show()
#
#     paretos = np.array(paretos)
#     print(paretos)
#     np.save('pareto_asr', paretos)


def pareto_model():
    x0 = 0
    x1 = 100
    num_samples = 30
    memory = 0
    step_sizes = np.geomspace(0.1, 0.7, 20)
    dim_state = 22
    dim_action = len(step_sizes)
    scaler = load("scaler_integ.pkl")
    integrator = Kronrod21()
    # integrator = IntegratorLinReg(step_sizes, load("linreg_integrator.bin"), integrator)

    env = IntegrationEnv(fun=DoublePendulumInteg(x0, x1), max_iterations=1000, initial_step_size=step_sizes[0],
                         error_tol=0.0005, memory=memory, nodes_per_integ=dim_state - 1, max_dist=x1 - x0)
    predictor = PredictorQ(step_sizes,
                           build_value_model(dim_state=dim_state, dim_action=dim_action,
                                             filename='predictor'),
                           scaler=scaler)

    performance = np.zeros((num_samples, 2))
    for i in range(num_samples):
        if i % 10 == 0:
            print(i)

        env.reset(integrator)
        _, evals, this_x1, err = integrate_env(predictor, integrator, env, x0, x1)

        performance[i, 0] = np.mean(err)
        performance[i, 1] = evals / this_x1

    print(np.mean(performance, axis=0))
    np.save('pareto_model2.npy', performance)


def pareto_const():
    x0 = 0
    x1 = 100
    num_samples = 20
    # step_sizes = np.geomspace(0.2, 0.27, 4)
    step_sizes = [0.222, 0.245, 0.265]
    integrator = Gauss21()

    paretos = []
    for step_size in step_sizes:
        print("step size: {}".format(step_size))
        errors = []
        fevals = []
        env = IntegrationEnv(fun=DoublePendulumInteg(x0, x1), max_iterations=1000, initial_step_size=step_size,
                             error_tol=0.0005, nodes_per_integ=integrator.num_nodes, max_dist=x1 - x0)
        predictor = PredictorConst(step_size)

        for i in range(num_samples):
            print(i)

            env.reset(integrator)
            _, evals, this_x1, err = integrate_env(predictor, integrator, env, x0, x1)

            errors.append(np.mean(env.errors))
            fevals.append(env.evals / x1)

        paretos.append((np.mean(errors), np.mean(fevals)))
        print("error: {}".format(np.mean(errors)))
        print("fevals: {}".format(np.mean(fevals)))

    paretos = np.array(paretos)
    np.save('pareto_const_g21.npy', paretos)


def pareto_quad():
    num_samples = 30
    x0 = 0
    x1 = 100
    tols = [2e-4, 5e-4, 8e-4, 3e-3]
    f = DoublePendulumInteg(x0, x1)
    limit = int((x1 - x0) * 50)

    # pre-cut the integral from x0 to x1 into subintervals before applying quad to each
    num_initial_intervals = 286

    initial_interval_length = (x1 - x0) / num_initial_intervals
    initial_interval_boundaries = []
    for i in range(num_initial_intervals):
        a = i * initial_interval_length
        b = (i + 1) * initial_interval_length
        initial_interval_boundaries.append((a, b))
    # print(f"initial intervals: {initial_interval_boundaries}")

    paretos = []
    for tol in tols:
        print("tol: {}".format(tol))
        errors = []
        fevals = []
        for i in range(num_samples):
            f.reset()
            step_errors = []
            step_fevals = []
            for a, b in initial_interval_boundaries:
                out = quad(f, a, b, epsrel=tol, limit=limit, full_output=True)
                info = out[2]
                # print(info)
                num_subintervals = info["last"]
                step_fevals.append(info["neval"] / (b - a))

                for j in range(num_subintervals):
                    a, b = info["alist"][j], info["blist"][j]
                    integ = info["rlist"][j]
                    step_errors.append(abs(integ - f.integral(a, b)))
                    # print(integ, f.integral(a, b))

            # print(step_errors)
            errors.append(np.mean(step_errors))
            fevals.append(np.mean(step_fevals))

        paretos.append((np.mean(errors), np.mean(fevals)))
        print("error: {}".format(np.mean(errors)))
        print("fevals: {}".format(np.mean(fevals)))

    paretos = np.array(paretos)
    np.save('pareto_precut_035.npy', paretos)


def pareto_clenshaw_curtis():
    """ doesnt work """
    num_samples = 1
    x0 = 0
    x1 = 100
    tols = [1e-9]
    f = DoublePendulumInteg(x0, x1)
    limit = int((x1 - x0) * 50)

    def prepare_fun(fun, a, b):
        def g(theta):
            x = np.cos(theta)
            return f(0.5 * (b - a) * x + 0.5 * (a + b)) * 0.5 * (b - a)
        return g

    paretos = []
    for tol in tols:
        print("tol: {}".format(tol))
        errors = []
        fevals = []
        for i in range(num_samples):
            f.reset()
            step_errors = []

            out = quad(prepare_fun(f, x0, x1), 0, np.pi,
                       weight='sin', wvar=1, full_output=True, limit=limit)
            info = out[2]
            print(f"total error {out[0] - f.integral(x0, x1)}")
            # print(info)
            num_subintervals = info["last"]

            for j in range(num_subintervals):
                a, b = info["alist"][j], info["blist"][j]
                integ = info["rlist"][j]
                # print(a, b)
                # translate a, b into original domain
                a = -1 + 2 * a / np. pi
                b = -1 + 2 * b / np. pi
                a = (x1 - x0) / 2 * a + (x0 + x1) / 2
                b = (x1 - x0) / 2 * b + (x0 + x1) / 2
                # print(a, b)
                step_errors.append(abs(integ - f.integral(a, b)))
                # print(integ, f.integral(a, b))

            # print(step_errors)
            errors.append(np.mean(step_errors))
            fevals.append(info["neval"] / (x1 - x0))

        paretos.append((np.mean(errors), np.mean(fevals)))
        print("error: {}".format(np.mean(errors)))
        print("fevals: {}".format(np.mean(fevals)))

    paretos = np.array(paretos)
    np.save('pareto_quad_cc.npy', paretos)


def plot_paretos():
    pareto_scipy_quad = np.load('pareto_quad.npy')
    pareto_scipy_quad2 = np.load('pareto_quad2.npy')
    pareto_scipy_quad_precut = np.load('pareto_quad_precut.npy')
    pareto_scipy_quad_precut2 = np.load('pareto_precut_035.npy')[[0, 2, 3]]
    pareto_constant = np.load('pareto_const.npy')
    pareto_constant_g21 = np.load('pareto_const_g21.npy')
    pareto_mod = np.load('pareto_model.npy')
    pareto_mod = np.mean(pareto_mod, axis=0)
    pareto_mod_g21 = np.load('pareto_model_g21.npy')
    pareto_mod_g21 = np.mean(pareto_mod_g21, axis=0)
    pareto_mod_optim = np.load('pareto_model_optim_weights.npy')
    pareto_mod_optim = np.mean(pareto_mod_optim, axis=0)
    # pareto_as = np.load('pareto_asr.npy')
    # pareto_mem1 = np.load('pareto_model_mem1.npy')
    # pareto_mem2 = np.load('pareto_model_mem2.npy')
    # pareto_rom = pareto_rom[np.argsort(pareto_rom[:, 0])]
    # pareto_mod_linreg = np.load('pareto_model_linreg.npy')
    # pareto_estim = np.load('pareto_model_estimator.npy')
    # pareto_quad = np.load('pareto_quad.npy')
    # pareto_linreg_estim = np.load('pareto_linreg_estim.npy')

    # fig = plt.figure(figsize=(5, 4), dpi=300)
    # # plt.xlim((1.5e-5, 2e-4))
    # # plt.ylim((70, 400))
    # plt.loglog(pareto_scipy_quad[:, 0], pareto_scipy_quad[:, 1], 'bx-', label='quad (GK21, subdivision)')
    # plt.loglog(pareto_scipy_quad2[:, 0], pareto_scipy_quad2[:, 1], 'bo-', label='quad2 (GK21, subdivision)')
    # plt.loglog(pareto_scipy_quad_precut[:, 0], pareto_scipy_quad_precut[:, 1], 'bs-', label='quad (GK21, subdivision, precut 1.0)')
    # plt.loglog(pareto_scipy_quad_precut2[:, 0], pareto_scipy_quad_precut2[:, 1], 'bv-', label='quad (GK21, subdivision, precut 0.5)')
    # # plt.loglog(pareto_scipy_quad_cc[:, 0], pareto_scipy_quad_cc[:, 1], 'mx-',
    # #            label='quad (Clenshaw-Curtis)')
    # plt.loglog(pareto_constant[:, 0], pareto_constant[:, 1], 'gx-', label='K21, const. step size')
    # plt.loglog(pareto_constant_g21[:, 0], pareto_constant_g21[:, 1], 'go-', label='G21, const. step size')
    #
    # plt.loglog(pareto_mod[0], pareto_mod[1], 'rv', label='model (K21)')
    # plt.loglog(pareto_mod_g21[0], pareto_mod_g21[1], 'rs', label='model (G21)')
    # plt.loglog(pareto_mod_optim[0], pareto_mod_optim[1], 'mo', label='model (optimized weights)')
    #
    # plt.legend(loc="upper right", framealpha=0.7)
    # plt.xlabel('error per step')
    # plt.ylabel('number of steps')
    # plt.grid(which='both')
    # plt.tight_layout()
    # plt.savefig('pareto.png')
    # plt.show()

    fig, ax = plot_pareto(pareto_mod, pareto_scipy_quad, pareto_scipy_quad_precut2, pareto_mod_g21, plot_scatter=False)
    ax.loglog(
        pareto_constant_g21[:, 0],
        pareto_constant_g21[:, 1],
        color="tab:blue",
        marker="+",
        label="G21 (const. step size)",
    )
    ax.loglog(
        pareto_mod_optim[0],
        pareto_mod_optim[1],
        color="tab:orange",
        marker="d",
        markersize=3,
        linestyle="None",
        label="Model (optim. weights)",
        alpha=0.8
    )
    ax.legend(
        loc="best",
        fontsize="xx-small",
        markerfirst=False,
        framealpha=0.6,
        ncol=1,
        columnspacing=0,
    )
    plt.savefig("pareto.pdf")
    plt.show()


def test_scipy_quad():
    # scipy.quad uses (G10, K21) Gauss-Kronrod quadrature
    x0 = -1
    x1 = 1
    f = BrokenPolynomial()
    print(f.integral(x0, x1))
    print('')
    for tol in [1e-8, 1e-2, 1, 10]:
        integral, abserr, infodict = quad(f, x0, x1, full_output=True, epsabs=tol)
        print(integral)
        print(abserr)
        print(infodict['neval'])
        print(infodict['last'])
        print(infodict['alist'])
        print(infodict['blist'])
        print(infodict['elist'])
        print('')


if __name__ == '__main__':
    # one_fun()
    # pareto_model()
    # pareto_const()
    # pareto_quad()
    # pareto_clenshaw_curtis()
    # pareto_romberg()
    plot_paretos()

