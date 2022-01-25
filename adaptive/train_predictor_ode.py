import numpy as np
import math
from matplotlib import pyplot as plt
from adaptive.environments import ODEEnv
from adaptive.integrator import ClassicRungeKutta, RKDP
from functions import Rotation, LorenzSystem, Pendulum, VanDerPol, HenonHeiles
from adaptive.experience import ExperienceODE
from adaptive.predictor import PredictorQODE
from adaptive.build_models import build_value_model, build_value_modelODE
from adaptive.performance_tracker import PerformanceTrackerODE
from pickle import load
from sklearn.preprocessing import StandardScaler
from copy import deepcopy


def main():
    gamma = 0.0  # discount factor for future rewards
    num_episodes = 100000
    step_sizes = [0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.56, 0.6]
    dim_state = 6  # nodes per integration step
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    x0 = np.array([0, 0.5, 0, 0])  # start point of integration
    d = x0.shape[0]  # dimension of the ODE state space

    # scale inputs of NN to have the order ~10^-1
    scaler = StandardScaler()
    scaler.mean_ = np.zeros((dim_state * d + 1) * (memory + 1))  # state = (h, k_1, ...)
    scaler.scale_ = np.ones((dim_state * d + 1) * (memory + 1))
    # scaler = load(open("test_scaler.pkl", "rb"))

    env = ODEEnv(fun=HenonHeiles(), max_iterations=10000, initial_step_size=step_sizes[0],
                 step_size_range=(step_sizes[0], step_sizes[-1]),
                 error_tol=0.00001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=100)
    experience = ExperienceODE(batch_size=64)

    predictor = PredictorQODE(step_sizes=step_sizes,
                              model=build_value_modelODE(dim_state=dim_state * d + 1, dim_action=dim_action,
                                                         filename='predictorODE', lr=0.001, memory=memory),
                              scaler=scaler)

    # integrator = ClassicRungeKutta()
    integrator = RKDP()

    perf_tracker = PerformanceTrackerODE(env, num_testfuns=1, t0=0, t1=20, integrator=integrator)

    for episode in range(num_episodes):
        state = env.reset(integrator=integrator)
        reward_total = 0
        loss_this_episode = 0
        steps = 0
        done = False
        eps = 0.75  # randomization
        print('episode: {}'.format(episode))

        while not done:
            # get action from actor
            actions = predictor.get_actions(state)
            action = choose_action(actions, eps, dim_action)

            # execute action
            next_state, reward, done, _ = env.iterate(predictor.action_to_stepsize(action), integrator)
            # print(next_state[0].f_evals)
            steps += 1
            reward_total += reward

            # learning
            action_next_state = predictor.get_actions(next_state)
            target = reward + gamma * np.max(action_next_state)
            target_actions = actions.squeeze()
            target_actions[action] = target

            # print(predictor.model(scaler.transform([state[0].flatten()])).numpy())
            # print(target_actions)
            # print('')

            experience.append(state=state, target=target_actions)
            if experience.is_full() or done:
                states, targets = experience.get_samples()
                # print(np.round(predictor.model(scaler.transform(states)).numpy() - actions, decimals=4))
                # print(actions)
                # print('')

                loss_predictor = predictor.train_on_batch(states, targets)
                loss_this_episode += loss_predictor
                experience.reset()

            state = next_state.copy()

        print('reward: {}'.format(reward_total))
        print('loss_predictor: {}'.format(loss_this_episode))

        if episode % 10 == 0 and episode > 0:
            perf_tracker.evaluate_performance(predictor)
            integrator = deepcopy(perf_tracker.integrator)
            perf_tracker.plot()
            perf_tracker.plot_pareto(num_points=7)
            perf_tracker.plot_best_models()
            perf_tracker.best_models.save()

        # if episode % 20 == 0:
        #     env.plot(episode=episode, t_min=0, t_max=2)
        if episode % 2 == 0:
            predictor.model.save_weights('predictorODE')


def choose_action(actions, eps, dim_action):
    """
    With probability 0.4*eps choose the action one above the favored action and with probability 0.4*eps the action
    below the favored. With propb 0.2*eps choose completely random.
    Otherwise choose the favored action.

    Parameters
    ----------
    actions : np.ndarray
    eps : float
    dim_action : int

    Returns
    -------
    int
    """
    favored = np.argmax(actions)
    rn = np.random.sample()

    if rn < 0.2 * eps:
        return np.random.randint(dim_action)
    if rn < 0.6 * eps:
        return min(favored + 1, dim_action - 1)
    if rn < eps:
        return max(favored - 1, 0)

    return favored


if __name__ == '__main__':
    # test()
    main()

