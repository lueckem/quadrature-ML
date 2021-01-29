import numpy as np
import math
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from adaptive.build_models import build_value_model
from adaptive.experience import Experience
from adaptive.environments import IntegrationEnv
from adaptive.predictor import PredictorQ
from adaptive.integrator import Simpson, IntegratorLinReg, Boole
from adaptive.performance_tracker import PerformanceTracker
from functions import Sinus, SuperposeSinus, BrokenPolynomial
from sklearn.externals.joblib import dump, load


def choose_action(actions, eps, dim_action):
    """
    Choose random action with probabilty eps. Otherwise choose what the predictor thinks is best.

    Parameters
    ----------
    actions : np.ndarray
    eps : float
    dim_action : int

    Returns
    -------
    int
    """
    if np.random.sample() < eps:
        return np.random.randint(0, dim_action - 1)
    else:
        return np.argmax(actions)


def choose_action2(actions, eps, dim_action):
    """
    With probability 0.5*eps choose the action one above the favored action and with probability 0.5*eps the action
    below the favored. Otherwise choose the favored action.

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
    if np.random.sample() < 0.5 * eps:
        return min(favored + 1, dim_action - 1)
    elif np.random.sample() < eps:
        return max(favored - 1, 0)
    else:
        return favored


def choose_action3(actions, eps, dim_action):
    """
    With probability (1-eps) choose the action with highest expected reward.
    Otherwise choose different action depending on expected reward.

    Parameters
    ----------
    actions : np.ndarray
    eps : float

    Returns
    -------
    int
    """
    sort_act = np.squeeze(np.argsort(-actions))
    dim_a = len(np.squeeze(actions))
    if dim_a == 2:
        probs = [1 - eps, eps]
    else:
        a = - 1 / (dim_a - 1) * np.log(0.01 / (1 - eps))
        probs = [(1 - eps) * np.exp(-a * idx) for idx in range(dim_a)]
        probs[1:dim_a - 1] = probs[1:dim_a - 1] / np.linalg.norm(probs[1:dim_a - 1], 1) * (eps - 0.01)

    return sort_act[np.random.choice(dim_a, p=probs)]


def main():
    gamma = 0.0
    num_episodes = 100000
    # step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]
    step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    # step_sizes = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4]
    # step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
    dim_state = 3  # nodes per integration step
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back

    # 7.5e-6
    env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.075,
                         error_tol=7.5e-6, nodes_per_integ=dim_state, memory=memory,
                         x0=-1, max_dist=2, step_size_range=(step_sizes[0], step_sizes[-1]))
    # env = IntegrationEnv(fun=Sinus(), max_iterations=128, initial_step_size=0.1, step_sizes=step_sizes,
    #                      error_tol=0.0005, nodes_per_integ=dim_state, memory=memory)
    experience = Experience(batch_size=32)

    predictor = PredictorQ(step_sizes=step_sizes,
                           model=build_value_model(dim_state=dim_state, dim_action=dim_action,
                                                   filename='predictor', lr=0.00001, memory=memory),
                           scaler=load('scaler.bin'))
    # integrator = IntegratorLinReg(step_sizes, load('linreg_models.bin'), load('scaler.bin'))
    # integrator = Boole()
    integrator = Simpson()

    perf_tracker = PerformanceTracker(env, num_testfuns=1000, x0=-1, x1=1)
    # losses = []
    # moving_average = []

    for episode in range(num_episodes):
        state = env.reset()
        reward_total = 0
        loss_this_episode = 0
        steps = 0
        done = False
        eps = 0.66

        if episode < 0:
            # eps = 0.01 + (1.0 - 0.01) * math.exp(-0.023 * episode
            eps = 0.2 + 0.8 * 2.71828 ** (-0.0146068 * episode)  # decrease from 1.0 to approx 0.2 at episode 300

        print('episode: {}'.format(episode))

        while not done:
            # get action from actor
            actions = predictor.get_actions(state)
            if episode < 0:
                action = choose_action(actions, eps, dim_action)
            else:
                action = choose_action3(actions, eps, dim_action)
            # print(actions.squeeze()[action])

            # execute action
            next_state, reward, done, _ = env.iterate(action, integrator)
            steps += 1
            reward_total += reward

            # learning
            action_next_state = predictor.get_actions(next_state)
            target = reward + gamma * np.max(action_next_state)
            target_actions = actions.squeeze()
            target_actions[action] = target
            # print(target)
            # print('')

            experience.append(state=state, target=target_actions)
            if experience.is_full() or done:
                states, targets = experience.get_samples()
                loss_predictor = predictor.train_on_batch(states, targets)
                loss_this_episode += loss_predictor
                experience.reset()

            state = next_state

        print('reward: {}'.format(reward_total))
        print('loss_predictor: {}'.format(loss_this_episode))

        # losses.append(loss_this_episode)
        # if episode % 10 == 0 and len(losses) > 99:
        #     moving_average.append(np.mean(losses[-100:]))
        #     plt.plot(moving_average, 'r')
        #     plt.pause(0.05)
        if episode % 100 == 0:
            perf_tracker.evaluate_performance(predictor, integrator)
            perf_tracker.plot()
            perf_tracker.plot_pareto(num_points=7)

        # if episode % 250 == 0:
        #     env.plot(episode=episode, x_min=-1.5, x_max=1.5)
        if episode % 10 == 0:
            predictor.model.save_weights('predictor')


def save_scaler():
    # step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    # step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]
    step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
    env = IntegrationEnv(fun=BrokenPolynomial(), max_iterations=256, initial_step_size=0.1,
                         step_sizes=step_sizes, error_tol=0.0005, memory=1, nodes_per_integ=3)

    # build Scaler
    scaler = StandardScaler()
    scaler.fit(env.sample_states(50000))
    dump(scaler, 'scaler_mem1.bin', compress=True)


def visualize_predictor():
    step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
    dim_state = 3
    dim_action = len(step_sizes)
    predictor = PredictorQ(model=build_value_model(dim_state=dim_state, dim_action=dim_action,
                                                   filename='predictor', lr=0.0001),
                           scaler=load('scaler.bin'))

    predictor.visualize([0.1, (-1.5, 1.5), (-1.5, 1.5)], step_sizes, flat=True)


if __name__ == '__main__':
    main()
    # save_scaler()
    # visualize_predictor()
