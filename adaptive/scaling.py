import numpy as np
from sklearn.preprocessing import StandardScaler
from adaptive.environments import ODEEnv
from adaptive.integrator import IntegratorODE
from pickle import dump
import random


def build_scaler(integrator, env, step_sizes, x0=None, num_episodes=1, filename=None):
    """
    Integrate env using random step sizes from step_sizes and use the encountered states to fit
    a scaler.

    Parameters
    ----------
    integrator : IntegratorODE
    env : ODEEnv
    step_sizes : list[float]
    x0 : list[np.ndarray], optional
        For every episode choose a random initial condition from x0.
    num_episodes : int, optional
    filename : str, optional
        If provided, the Standardscaler is saved under "filename.pkl".

    Returns
    -------
    StandardScaler
    """

    if x0 is None:
        x0 = [env.x0]

    states = []
    for episode in range(num_episodes):
        env.x0 = random.choice(x0)
        state = env.reset(integrator)
        states.append(state[0].flatten())
        done = False
        while not done:
            step_size = random.choice(step_sizes)
            next_state, _, done, _ = env.iterate(step_size, integrator)
            states.append(next_state[0].flatten())

    scaler = StandardScaler().fit(states)

    if filename is not None:
        dump(scaler, open("{}.pkl".format(filename), "wb"))

    return scaler


def test_build_scaler():
    from functions import HenonHeiles
    from adaptive.integrator import RKDP

    step_sizes = [0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.56, 0.6]
    dim_state = 6
    x0 = np.array([0, 0.5, 0, 0])
    integrator = RKDP()
    env = ODEEnv(fun=HenonHeiles(), max_iterations=10000, initial_step_size=step_sizes[0],
                 step_size_range=(step_sizes[0], step_sizes[-1]),
                 error_tol=0.00001, nodes_per_integ=dim_state, x0=x0, max_dist=100)

    scaler = build_scaler(integrator, env, step_sizes, filename="test_scaler")
    print(scaler.mean_)
    print(scaler.scale_)


if __name__ == '__main__':
    test_build_scaler()
