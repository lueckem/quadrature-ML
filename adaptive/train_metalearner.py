import numpy as np
import math
from matplotlib import pyplot as plt
from adaptive.environments import ODEEnv
from adaptive.integrator import ClassicRungeKutta, RKDP
from functions import Rotation, LorenzSystem, Pendulum
from adaptive.experience import ExperienceODE
from adaptive.predictor import PredictorQODE, PredictorConstODE, MetaQODE
from adaptive.build_models import build_value_model, build_value_modelODE
from adaptive.performance_tracker import PerformanceTrackerODE
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


def stepsize_to_idx(h):
    if h > 0.2:
        return -1
    if h > 0.075:
        return -0.5
    if h > 0.025:
        return 0
    if h > 0.0075:
        return 0.5
    return 1


def main():
    gamma = 0.0
    num_episodes = 100000
    d = 2  # dimension of the ODE state space
    x0 = np.array([1, 1])
    basis_learners = []

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
                                                      filename='metaODE', memory=memory, lr=0.01),
                           scaler=scaler, use_idx=False)

    # define environment
    env = ODEEnv(fun=Pendulum(switchpoints=(0.05, 3.3)), max_iterations=10000, initial_step_size=0.25,
                 step_size_range=(0.005, 0.48),
                 error_tol=0.00001, nodes_per_integ=dim_state, memory=memory, x0=x0, max_dist=50,
                 stepsize_to_idx=stepsize_to_idx)
    experience = ExperienceODE(batch_size=64)

    integrator = RKDP()
    perf_tracker = PerformanceTrackerODE(env, num_testfuns=1, t0=0, t1=50, integrator=integrator)

    for episode in range(num_episodes):
        state = env.reset(integrator=integrator)

        reward_total = 0
        loss_this_episode = 0
        steps = 0
        done = False
        eps = 0.2
        print('episode: {}'.format(episode))

        while not done:
            # get action from actor
            actions = metalearner.get_actions(state)
            action = choose_action(actions, eps)

            # execute action
            next_state, reward, done, _ = env.iterate(metalearner.action_to_stepsize(action, state), integrator)
            # print(next_state[0].f_evals)
            steps += 1
            reward_total += reward

            # learning
            action_next_state = metalearner.get_actions(next_state)
            target = reward + gamma * np.max(action_next_state)
            target_actions = actions.squeeze()
            target_actions[action] = target

            # print(predictor.model(scaler.transform([state[0].flatten()])).numpy())
            # print(target_actions)
            # print('')

            experience.append(state=state, target=target_actions)
            if experience.is_full() or done:
                states, targets = experience.get_samples(use_idx=metalearner.use_idx)
                # print(np.round(predictor.model(scaler.transform(states)).numpy() - actions, decimals=4))
                # print(actions)
                # print('')

                loss_predictor = metalearner.train_on_batch(states, actions)
                loss_this_episode += loss_predictor
                experience.reset()

            state = next_state.copy()

        print('reward: {}'.format(reward_total))
        print('loss_predictor: {}'.format(loss_this_episode))

        if episode % 10 == 0:
            perf_tracker.evaluate_performance(metalearner, integrator)
            perf_tracker.plot()
            perf_tracker.plot_pareto(num_points=7)

        # if episode % 20 == 0:
        #     env.plot(episode=episode, t_min=0, t_max=2)
        if episode % 2 == 0:
            metalearner.model.save_weights('metaODE')


def choose_action(actions, eps):
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


if __name__ == '__main__':
    # test()
    main()