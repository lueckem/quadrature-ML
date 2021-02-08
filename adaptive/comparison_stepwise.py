import numpy as np
import math
from adaptive.environments import IntegrationEnv
from adaptive.build_models import build_value_model, build_estimator_model
from functions import Sinus, Pulse, StaggeredSinus, SuperposeSinus, Polynomial, BrokenPolynomial
from joblib import dump, load
from scipy.integrate import quad, romberg
from adaptive.adapt_simps import AdaptSimps, AdaptSimpsConstEvals, Simps
from adaptive.boole import BoolesRule
from adaptive.romberg import Romberg
from adaptive.predictor import *
from adaptive.comparison import integrate_env
from adaptive.integrator import Simpson, IntegratorLinReg, Boole
from adaptive.error_estimator import Estimator


def one_fun():
    x0 = -1
    x1 = 1
    # step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
    dim_state = 3
    dim_action = len(step_sizes)
    memory = 1
    env = IntegrationEnv(fun=BrokenPolynomial(), max_iterations=256, initial_step_size=0.075,
                         step_sizes=step_sizes, error_tol=7.5e-6, memory=memory)
    predictor = PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor', memory=memory),
                           load('scaler_mem1.bin'))
    # integrator = IntegratorLinReg(step_sizes, load('linreg_models.bin'), load('scaler.bin'))
    integrator = Simpson()

    # _, evals, _, errors = integrate_env(predictor, Simpson(), env, x0, x1, plot=True)
    # print('new x1: {}'.format(x1))
    # print('Predictor error total: {}'.format(np.sum(errors)))
    # print('Predictor error per step: {}'.format(np.mean(errors)))
    # print('Predictor evals: {}'.format(evals))
    # print('')

    env.reset(reset_params=False)
    _, evals, x1, errors = integrate_env(predictor, integrator, env, x0, x1, plot=True)
    print('new x1: {}'.format(x1))
    print('Predictor error total: {}'.format(np.sum(errors)))
    print('Predictor error per step: {}'.format(np.mean(errors)))
    print('Predictor evals: {}'.format(evals))
    print('')

    env.reset(reset_params=False)
    asr = AdaptSimpsConstEvals(env.fun, x0, x1)
    asr(200)
    errors = asr.stepwise_errors
    print('ASR error total: {}'.format(np.sum(errors)))
    print('ASR error per step: {}'.format(np.mean(errors)))
    print('ASR evals: {}'.format(asr.evals))
    print('')
    asr.plot()

    env.reset(reset_params=False)
    simps = Simps(env.fun, x0, x1)
    integ_simps, errors = simps(num_evals=evals, stepwise_error=True)
    # integ_simps, errors = simps(step_size=0.11, stepwise_error=True)
    print('Simpson error total: {}'.format(np.sum(errors)))
    print('Simpson error per step: {}'.format(np.mean(errors)))
    print('Simpson evals: {}'.format(simps.evals))
    print('')
    simps.plot()

    env.reset(reset_params=False)
    rom = Romberg(env.fun, x0, x1, tol=0.0005, order=2)
    integ, errors = rom(0.15, stepwise_errors=True)
    print('Romberg error total: {}'.format(np.sum(errors)))
    print('Romberg error per step: {}'.format(np.mean(errors)))
    print('Romberg evals: {}'.format(rom.evals))
    rom.plot()


def one_fun_boole():
    x0 = 0
    x1 = 10
    step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    dim_state = 5
    dim_action = len(step_sizes)
    memory = 1
    env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.1,
                         step_sizes=step_sizes, error_tol=0.000001, memory=memory, nodes_per_integ=dim_state)
    predictor = PredictorQ(
        build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor', memory=memory),
        load('scaler_boole_mem1.bin'))
    # integrator = IntegratorLinReg(step_sizes, load('linreg_models.bin'), load('scaler.bin'))
    integrator = Boole()

    _, evals, x1, errors = integrate_env(predictor, integrator, env, x0, x1, plot=True)
    print('new x1: {}'.format(x1))
    print('Predictor error total: {}'.format(np.sum(errors)))
    print('Predictor error per step: {}'.format(np.mean(errors)))
    print('Predictor evals: {}'.format(evals))
    print('')

    env.reset(reset_params=False)
    booles = BoolesRule(env.fun, x0, x1)
    integ_simps, errors = booles(num_evals=evals, stepwise_error=True)
    print('Boole error total: {}'.format(np.sum(errors)))
    print('Boole error per step: {}'.format(np.mean(errors)))
    print('Boole evals: {}'.format(booles.evals))
    print('')
    booles.plot()

    env.reset(reset_params=False)
    simps = Simps(env.fun, x0, x1)
    integ_simps, errors = simps(num_evals=evals, stepwise_error=True)
    print('Simpson error total: {}'.format(np.sum(errors)))
    print('Simpson error per step: {}'.format(np.mean(errors)))
    print('Simpson evals: {}'.format(simps.evals))
    print('')
    simps.plot()

    env.reset(reset_params=False)
    rom = Romberg(env.fun, x0, x1, tol=0.0005, order=3)
    integ, errors = rom(0.15, stepwise_errors=True)
    print('Romberg error total: {}'.format(np.sum(errors)))
    print('Romberg error per step: {}'.format(np.mean(errors)))
    print('Romberg evals: {}'.format(rom.evals))
    rom.plot()


def compare_simps_tol():
    """ compare model to simpson rule, stepsize for simps is adapted to match tol. """
    x0 = 0.0
    num_samples = 2000
    error_predictor = []
    error_simps = []
    evals_simps = []
    evals_predictor = []

    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    dim_state = 3
    dim_action = len(step_sizes)
    env = IntegrationEnv(fun=SuperposeSinus(5), max_iterations=256, initial_step_size=0.15,
                         step_sizes=step_sizes, error_tol=0.0005)
    predictor = PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
                           load('scaler.bin'))

    for i in range(num_samples):
        if i % 10 == 0:
            print(i)

        x1 = 20.0

        # model
        env.reset()
        _, evals, x1, errors = integrate_env(predictor, Simpson(), env, x0, x1)
        error_pred_step = np.mean(errors)
        env.reset(reset_params=False)

        # simpson
        step_size = 0.177
        simp = Simps(env.fun, x0, x1)
        _, error_simps_step = simp(step_size=step_size, stepwise_error=True)
        error_simps_step = np.mean(error_simps_step)
        evals_simps_step = simp.evals
        # for j in range(10):
        #     simp = Simps(env.fun, x0, x1)
        #     integ_rom, error_simps_step = simp(step_size=step_size, stepwise_error=True)
        #     error_simps_step /= (simp.evals - 1.0) / 2.0
        #     evals_simps_step = simp.evals
        #     if error_simps_step < 0.0001:
        #         step_size *= 1.5
        #     elif error_simps_step > 0.0005:
        #         step_size /= 2.0
        #     else:
        #         break

        error_predictor.append(error_pred_step)
        error_simps.append(error_simps_step)
        evals_predictor.append(evals)
        evals_simps.append(evals_simps_step)

    error_simps = np.array(error_simps)
    not_converged = np.concatenate((error_simps[error_simps > 0.0005], error_simps[error_simps < 0.0001]))
    if len(not_converged > 0):
        print('simps did not converge in some cases:')
        print(not_converged)

    mean_error_predictor = np.mean(error_predictor)
    var_error_predictor = np.var(error_predictor)

    mean_error_simps = np.mean(error_simps)
    var_error_simps = np.var(error_simps)

    mean_evals_predictor = np.mean(evals_predictor)
    var_evals_predictor = np.var(evals_predictor)

    mean_evals_simps = np.mean(evals_simps)
    var_evals_simps = np.var(evals_simps)

    print('Avg. predictor number of function evaluations per episode: {}'.format(mean_evals_predictor))
    print('Avg. predictor error per step: {}'.format(mean_error_predictor))
    print('Avg. simpson number of function evaluations per episode: {}'.format(mean_evals_simps))
    print('Avg. simpson error per step: {}'.format(mean_error_simps))
    print('')
    print('Variance of predictor number of function evaluations per episode: {}'.format(var_evals_predictor))
    print('Variance of predictor error per step: {}'.format(var_error_predictor))
    print('Variance of simpson number of function evaluations per episode: {}'.format(var_evals_simps))
    print('Variance of simpson error per step: {}'.format(var_error_simps))


def compare_romberg():
    x0 = 0.0
    num_samples = 100
    error_predictor = []
    error_rom = []
    evals_rom = []
    evals_predictor = []

    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    dim_state = 3
    dim_action = len(step_sizes)
    env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.15,
                         step_sizes=step_sizes, error_tol=0.0005)
    predictor = PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor'),
                           load('scaler.bin'))

    for i in range(num_samples):
        if i % 10 == 0:
            print(i)

        x1 = 20.0

        # model
        env.reset()
        _, evals, x1, errors = integrate_env(predictor, Simpson(), env, x0, x1)
        error_pred_step = np.mean(errors)
        env.reset(reset_params=False)

        # romberg
        tol = 0.0003

        rom = Romberg(env.fun, x0, x1, tol=tol, order=2)
        integ_rom, errors = rom(0.15, True)
        error_rom_step = np.mean(errors)
        evals_rom_step = rom.evals

        # for j in range(10):
        #     rom = Romberg(env.fun, x0, x1, tol=tol, order=2)
        #     integ_rom, errors = rom(0.15, True)
        #     error_rom_step = np.mean(errors)
        #     evals_rom_step = rom.evals
        #     if error_rom_step < 0.0001:
        #         tol *= 2.0
        #     elif error_rom_step > 0.0005:
        #         tol /= 3.0
        #     else:
        #         break

        error_predictor.append(error_pred_step)
        error_rom.append(error_rom_step)
        evals_predictor.append(evals)
        evals_rom.append(evals_rom_step)

    # error_rom = np.array(error_rom)
    # not_converged = np.concatenate((error_rom[error_rom > 0.0005], error_rom[error_rom < 0.0001]))
    # if len(not_converged > 0):
    #     print('romberg did not converge in some cases:')
    #     print(not_converged)

    mean_error_predictor = np.mean(error_predictor)
    var_error_predictor = np.var(error_predictor)

    mean_error_rom = np.mean(error_rom)
    var_error_rom = np.var(error_rom)

    mean_evals_predictor = np.mean(evals_predictor)
    var_evals_predictor = np.var(evals_predictor)

    mean_evals_rom = np.mean(evals_rom)
    var_evals_rom = np.var(evals_rom)

    print('Avg. predictor number of function evaluations per episode: {}'.format(mean_evals_predictor))
    print('Avg. predictor error per step: {}'.format(mean_error_predictor))
    print('Avg. romberg number of function evaluations per episode: {}'.format(mean_evals_rom))
    print('Avg. romberg error per step: {}'.format(mean_error_rom))
    print('')
    print('Variance of predictor number of function evaluations per episode: {}'.format(var_evals_predictor))
    print('Variance of predictor error per step: {}'.format(var_error_predictor))
    print('Variance of rom number of function evaluations per episode: {}'.format(var_evals_rom))
    print('Variance of rom error per step: {}'.format(var_error_rom))


def pareto_simpson():
    """ Find pairs (mean error, avg steps) for equidistand simpson rules with different step sizes."""
    x0 = -1
    x1 = 1
    num_samples = 5000
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]

    env = IntegrationEnv(fun=BrokenPolynomial(), max_iterations=1000, initial_step_size=0.1,
                         step_sizes=step_sizes, error_tol=0.0005)
    step_sizes = np.linspace(0.05, 0.4, 16)
    paretos = []

    for step_size in step_sizes:
        print(step_size)
        this_errors = []
        for _ in range(num_samples):
            env.reset()
            simp = Simps(env.fun, x0, x1)
            _, error_simps_step = simp(step_size=step_size, stepwise_error=True)
            this_errors.append(np.mean(error_simps_step))
        paretos.append((np.mean(this_errors), (x1 - x0) / step_size + 1))

        # plt.hist(this_errors, 25)
        # plt.show()

    paretos = np.array(paretos)
    np.save('pareto_simpson', paretos)


def pareto_boole():
    """ Find pairs (mean error, avg steps) for equidistand booles rules with different step sizes."""
    x0 = 0.0
    x1 = 20.0
    num_samples = 4000
    step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]

    env = IntegrationEnv(fun=SuperposeSinus(5), max_iterations=256, initial_step_size=0.05,
                         step_sizes=step_sizes, error_tol=0.0005, memory=0)
    step_sizes = np.linspace(0.025, 0.25, 31)
    paretos = []

    for step_size in step_sizes:
        print(step_size)
        this_errors = []
        for _ in range(num_samples):
            env.reset()
            simp = BoolesRule(env.fun, x0, x1)
            _, error_simps_step = simp(step_size=step_size, stepwise_error=True)
            this_errors.append(np.mean(error_simps_step))
        paretos.append((np.mean(this_errors), (x1 - x0) / step_size + 1))

    paretos = np.array(paretos)
    np.save('pareto_boole', paretos)


def pareto_asr():
    x0 = 0
    x1 = 20
    num_samples = 5000
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]

    env = IntegrationEnv(fun=SuperposeSinus(5), max_iterations=1000, initial_step_size=0.1,
                         step_sizes=step_sizes, error_tol=0.0005)
    nevals = [75, 90, 108, 130, 155, 186, 223, 268, 322, 386]
    paretos = []

    for neval in nevals:
        print(neval)
        this_errors = []
        this_evals = []
        for _ in range(num_samples):
            env.reset()
            asr = AdaptSimpsConstEvals(env.fun, x0, x1)
            asr(neval)
            stepwise_errors = asr.stepwise_errors
            this_errors.append(np.mean(stepwise_errors))
            this_evals.append(asr.evals)
        paretos.append((np.mean(this_errors), np.mean(this_evals)))

        # plt.hist(this_errors, 25)
        # plt.show()

    paretos = np.array(paretos)
    print(paretos)
    np.save('pareto_asr', paretos)


def pareto_model():
    x0 = -1
    x1 = 1
    num_samples = 500
    memory = 1

    # step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    # step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
    dim_state = 3
    dim_action = len(step_sizes)
    env = IntegrationEnv(fun=BrokenPolynomial(), max_iterations=1000, initial_step_size=0.075,
                         step_sizes=step_sizes, error_tol=0.0005, memory=memory, nodes_per_integ=dim_state)
    predictor = PredictorQ(build_value_model(dim_state=dim_state, dim_action=dim_action, filename='predictor', memory=memory),
                           load('scaler_mem1.bin'))

    # integrator = IntegratorLinReg(step_sizes, load('linreg_models_estim.bin'))
    integrator = Simpson()
    # estimator = Estimator(build_estimator_model(dim_state, lr=0.0001, filename='estimator'), load('scaler.bin'),
    #                       threshold=100 * 7.5e-6)

    errors = []
    steps = []
    x1s = []
    for i in range(num_samples):
        if i % 10 == 0:
            print(i)

        env.reset()
        _, evals, this_x1, err = integrate_env(predictor, integrator, env, x0, x1)
        errors.append(np.mean(err))
        steps.append(evals)
        x1s.append(this_x1)

    print(np.mean(steps))
    print(np.mean(errors))
    print(np.mean(x1s))
    print(np.quantile(errors, 0.9))
    np.save('pareto_model_mem1.npy', np.array([np.mean(errors), np.mean(steps)]))

    plt.hist(errors, 25)
    plt.show()


def pareto_romberg():
    x0 = -1
    x1 = 1
    num_samples = 5000
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]

    env = IntegrationEnv(fun=BrokenPolynomial(), max_iterations=256, initial_step_size=0.075,
                         step_sizes=step_sizes, error_tol=0.0005)
    tolerances = [1e-5, 3e-4, 1e-4, 7e-3, 4e-3, 1e-3, 7e-2, 1e-2]
    paretos = []

    for tol in tolerances:
        print(tol)
        this_errors = []
        this_step_sizes = []
        for _ in range(num_samples):
            env.reset()
            rom = Romberg(env.fun, x0, x1, tol=tol, order=2)
            _, error_rom_steps = rom(env.initial_step_size, stepwise_errors=True)
            this_errors.append(np.mean(error_rom_steps))
            this_step_sizes.append(rom.evals)
        paretos.append((np.mean(this_errors), np.mean(this_step_sizes)))

    paretos = np.array(paretos)
    print(paretos)
    np.save('pareto_romberg', paretos)


def pareto_quad():
    x0 = -1
    x1 = 1
    num_samples = 1000
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]

    env = IntegrationEnv(fun=BrokenPolynomial(), max_iterations=256, initial_step_size=0.075,
                         step_sizes=step_sizes, error_tol=0.0005)
    tolerances = [100, 1, 1e-1]
    paretos = []

    for tol in tolerances:
        print(tol)
        this_errors = []
        this_step_sizes = []
        for _ in range(num_samples):
            env.reset()
            integral, abserr, infodict = quad(env.fun, x0, x1, full_output=True, epsabs=tol)
            evals = infodict['neval']
            err = abs(env.fun.integral(x0, x1) - integral) / evals
            this_errors.append(err)
            this_step_sizes.append(evals)
        paretos.append((np.mean(this_errors), np.mean(this_step_sizes)))

    paretos = np.array(paretos)
    print(paretos)
    np.save('pareto_quad', paretos)


def plot_pareto():
    pareto_simps = np.load('pareto_simpson.npy')
    pareto_mod = np.load('pareto_model.npy')
    # pareto_as = np.load('pareto_asr.npy')
    # pareto_mem1 = np.load('pareto_model_mem1.npy')
    # pareto_mem2 = np.load('pareto_model_mem2.npy')
    # pareto_rom = pareto_rom[np.argsort(pareto_rom[:, 0])]
    # pareto_mod_linreg = np.load('pareto_model_linreg.npy')
    # pareto_estim = np.load('pareto_model_estimator.npy')
    # pareto_quad = np.load('pareto_quad.npy')
    # pareto_linreg_estim = np.load('pareto_linreg_estim.npy')

    fig = plt.figure(figsize=(5, 4), dpi=300)
    # plt.xlim((1.5e-5, 2e-4))
    # plt.ylim((70, 400))
    plt.loglog(pareto_simps[:, 0], pareto_simps[:, 1], 'bx-', label='equidistant')
    # plt.loglog(pareto_as[:, 0], pareto_as[:, 1], 'g+-', label='subdivision')
    plt.loglog(pareto_mod[0], pareto_mod[1], 'rv', label='m=0')
    # plt.loglog(pareto_mod_linreg[0], pareto_mod_linreg[1], 'ro', label='m=0, optim. weights')
    # plt.loglog(pareto_mem1[0], pareto_mem1[1], 'ks', label='m=1')
    # plt.loglog(pareto_mem2[0], pareto_mem2[1], 'cx', label='mem2')
    # plt.loglog(pareto_rom[:, 0], pareto_rom[:, 1], 'gx-', label='adapt. romberg (order=2)')
    # plt.loglog(pareto_estim[0], pareto_estim[1], 'rs', label='model + error estimator')
    # plt.loglog(pareto_quad[:, 0], pareto_quad[:, 1], 'kx-', label='quad')
    # plt.loglog(pareto_linreg_estim[0], pareto_linreg_estim[1], 'rX', label='optimized_weights + error estimator')

    plt.legend()
    plt.xlabel('error per step')
    plt.ylabel('number of steps')
    plt.grid(which='both')
    plt.tight_layout()
    plt.savefig('pareto.png')
    plt.show()


def test_scipy_quad():
    # scipy.quad uses (G10, K21) Gauss-Kronrod quadrature, so each step has 21 evaluations and the next step-size
    # is estimated via the error-estimate abs(G10 - K21)
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
    # compare()
    one_fun()
    # one_fun_boole()
    # compare_romberg()
    # pareto_simpson()
    # pareto_asr()
    # pareto_model()
    # pareto_romberg()
    # pareto_quad()
    # plot_pareto()
    # test_scipy_quad()

    # pareto_mod = np.load('pareto_model_mem1.npy')
    # pareto_o = np.load('pareto_model_linreg.npy')
    # print(pareto_mod)
    # pareto_mod[0] = 1.1892e-04
    # pareto_o[1] = 88.47
    # pareto_mod[1] = 88.47
    # print(pareto_o)
    # np.save('pareto_model_mem1.npy', pareto_mod)
    # np.save('pareto_model_linreg.npy', pareto_o)

    # [1.3523636e-04 8.6474500e+01]

