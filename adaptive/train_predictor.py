import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from adaptive.build_models import build_value_model
from adaptive.experience import Experience
from adaptive.environments import IntegrationEnv
from adaptive.predictor import PredictorQ
from adaptive.integrator import Simpson, IntegratorLinReg, Boole, Kronrod21
from adaptive.performance_tracker import PerformanceTracker
from functions import Sinus, SuperposeSinus, BrokenPolynomial, Pulse, DoublePendulumInteg
from joblib import dump, load


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
    step_sizes = [0.25, 0.29, 0.34, 0.4, 0.46, 0.54, 0.63, 0.73, 0.85, 1]
    step_sizes = np.geomspace(0.1, 0.7, 20)
    error_tol = 1e-7
    dim_action = len(step_sizes)
    memory = 0  # how many integration steps the predictor can look back
    integrator = Kronrod21()
    dim_state = integrator.num_nodes + 1
    x0 = 0
    x1 = 100

    scaler = load("scaler_integ.pkl")
    env = IntegrationEnv(fun=DoublePendulumInteg(x0, x1), max_iterations=10000, initial_step_size=step_sizes[0],
                         error_tol=error_tol, nodes_per_integ=dim_state - 1, memory=memory,
                         x0=x0, max_dist=x1 - x0, step_size_range=(step_sizes[0], step_sizes[-1]))
    experience = Experience(batch_size=32)

    predictor = PredictorQ(step_sizes=step_sizes,
                           model=build_value_model(dim_state=dim_state, dim_action=dim_action,
                                                   filename='predictor', lr=0.001, memory=memory),
                           scaler=scaler)

    perf_tracker = PerformanceTracker(env, num_testfuns=2, x0=0, x1=50, integrator=integrator)

    for episode in range(num_episodes):
        state = env.reset(integrator)
        reward_total = 0
        loss_this_episode = 0
        steps = 0
        done = False
        eps = 0.66

        print('episode: {}'.format(episode))

        while not done:
            # get action from actor
            actions = predictor.get_actions(state)
            action = choose_action2(actions, eps, dim_action)
            step_size = predictor.action_to_stepsize(action)

            # execute action
            next_state, reward, done, _ = env.iterate(step_size, integrator)
            steps += 1
            reward_total += reward

            # learning
            action_next_state = predictor.get_actions(next_state)
            target = reward + gamma * np.max(action_next_state)
            target_actions = actions.squeeze()
            target_actions[action] = target
            # print(target)
            # print('')

            experience.append(state=state[0], target=target_actions)
            if experience.is_full() or done:
                states, targets = experience.get_samples()
                loss_predictor = predictor.train_on_batch(states, targets)
                loss_this_episode += loss_predictor
                experience.reset()

            state = next_state

        print('reward: {}'.format(reward_total))
        print('loss_predictor: {}'.format(loss_this_episode))

        if episode % 10 == 0:
            perf_tracker.evaluate_performance(predictor, integrator)
            perf_tracker.plot()
            perf_tracker.plot_pareto(num_points=7)

        if episode % 2 == 0:
            predictor.model.save_weights('predictor')


# def save_scaler():
#     # step_sizes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
#     # step_sizes = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4]
#     step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
#     env = IntegrationEnv(fun=Sinus(), max_iterations=256, initial_step_size=0.075,
#                          error_tol=7.5e-6, nodes_per_integ=3, memory=1,
#                          x0=-1, max_dist=2, step_size_range=(step_sizes[0], step_sizes[-1]))
#
#     # build Scaler
#     scaler = StandardScaler()
#     scaler.fit(env.sample_states(50000))
#     dump(scaler, 'scaler_mem1.bin', compress=True)


# def visualize_predictor():
#     step_sizes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.67]
#     dim_state = 3
#     dim_action = len(step_sizes)
#     predictor = PredictorQ(step_sizes=step_sizes,
#                            model=build_value_model(dim_state=dim_state, dim_action=dim_action,
#                                                    filename='predictor', lr=0.0001),
#                            scaler=load('scaler.bin'))
#
#     predictor.visualize([0.1, (-1.5, 1.5), (-1.5, 1.5)], step_sizes, flat=True)


if __name__ == '__main__':
    main()
    # save_scaler()
    # visualize_predictor()
