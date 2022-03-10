import numpy as np
from sklearn.preprocessing import StandardScaler
from adaptive.integrator import StateODE


class Predictor:
    def __call__(self, state):
        """
        Base Predictor call.

        Parameters
        ----------
        state : list[np.ndarray]

        Returns
        -------
        action: int
            the next step size should be step_sizes[action]
        """
        return 0


class PredictorQ(Predictor):
    def __init__(self, step_sizes, model, scaler):
        """
        Predictor using a neural network model.

        Parameters
        ----------
        step_sizes : list[float]
        model : tf.keras.Model
        scaler : StandardScaler
        """
        self.step_sizes = step_sizes
        self.scaler = scaler
        self.model = model

    def __call__(self, state):
        """
        Parameters
        ----------
        state : list[np.ndarray]

        Returns
        -------
        float
            the next step size
        """
        actions = self.model(self.scaler.transform([state[0]]))
        action = np.argmax(actions)
        return self.step_sizes[action]

    def get_actions(self, state):
        """
        Return the value of each possible action.

        Parameters
        ----------
        state : list[np.ndarray]

        Returns
        -------
        np.ndarray
        """
        return self.model(self.scaler.transform([state[0]])).numpy()

    def action_to_stepsize(self, action):
        """

        Parameters
        ----------
        action : int

        Returns
        -------
        float
            step_sizes[action]
        """
        return self.step_sizes[action]

    def train_on_batch(self, states, actions):
        """
        Train the model.

        Parameters
        ----------
        states : np.ndarray
        actions : np.ndarray
        """
        return self.model.train_on_batch(self.scaler.transform(states), actions)

    # def visualize(self, domains, step_sizes, flat=False):
    #     """
    #     Visualize the predicting function in a 3D surface plot.
    #
    #     Parameters
    #     ----------
    #     domains : List[int or Tuple[int]]
    #         domains for visualization, e.g. if the state is 5-d [0.1, (-1,1), 0, (-1,1), 0.2] would result in a plot
    #         where h = 0.1, f3-f1 = 0, f5-f1=0.2 and (f2-f1), (f4-f1) vary between -1 and 1.
    #         As the plot is 3D, only 2 domains should be tuples.
    #     step_sizes : List[float]
    #     flat : bool, optional
    #         whether to plot as a surface plot or as a flat heatmap
    #     """
    #     # find idx of tuples in domains
    #     dom_idx = []
    #     for idx, domain in enumerate(domains):
    #         if isinstance(domain, tuple) and len(domain) == 2:
    #             dom_idx.append(idx)
    #     if len(dom_idx) != 2:
    #         raise ValueError("Need exactly 2 tuples of length 2 in domains.")
    #
    #     # build data
    #     X = np.linspace(domains[dom_idx[0]][0], domains[dom_idx[0]][1], 61)
    #     Y = np.linspace(domains[dom_idx[1]][0], domains[dom_idx[1]][1], 61)
    #     X, Y = np.meshgrid(X, Y)
    #
    #     outputs = np.zeros(X.shape)
    #     state = domains
    #     state[dom_idx[0]] = 0
    #     state[dom_idx[1]] = 0
    #     state = np.array(state)
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[1]):
    #             state[dom_idx[0]] = X[i, j]
    #             state[dom_idx[1]] = Y[i, j]
    #             outputs[i, j] = step_sizes[self.__call__(state)]
    #
    #     # plot
    #     if not flat:
    #         fig = plt.figure()
    #         ax = fig.gca(projection='3d')
    #         surf = ax.plot_surface(X, Y, outputs, cmap=cm.viridis, linewidth=0)
    #         ax.set_zlabel('suggested stepsize')
    #     else:
    #         fig, ax = plt.subplots()
    #         surf = ax.pcolormesh(X, Y, outputs, cmap=cm.viridis)
    #
    #     cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    #     ax.set_xlabel('f2 - f1')
    #     ax.set_ylabel('f3 - f1')
    #     plt.grid()
    #     plt.title('input stepsize: {}'.format(domains[0]))
    #     plt.show()


class PredictorConst(Predictor):
    def __init__(self, c):
        self.c = c

    def __call__(self, state):
        return self.c


class PredictorRandom(Predictor):
    def __init__(self, step_sizes):
        self.step_sizes = step_sizes

    def __call__(self, state):
        return np.random.choice(self.step_sizes)


class PredictorODE:
    def __call__(self, states):
        """
        Base Predictor call for ODE.

        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        float
        """
        return 0.1


class PredictorQODE(PredictorODE):
    def __init__(self, step_sizes, model, scaler, use_idx=False):
        """
        Predictor using a neural network model.

        Parameters
        ----------
        step_sizes : list[float]
        model : tf.keras.Model
        scaler : StandardScaler
        use_idx : bool, optional
            whether the step size is used in the state or a idx referring to it
            (e.g. if the step sizes are from different orders like [0.0001, 0.0002, 0.1], idx might work better)
        """
        self.use_idx = use_idx
        self.step_sizes = step_sizes
        self.scaler = scaler
        self.model = model

    def __call__(self, states, eps=0):
        """
        Parameters
        ----------
        states : list[StateODE]
        eps : float
            probability that a random action is chosen instead of the one with highest value

        Returns
        -------
        float
            step_sizes[action]
        """
        states = np.concatenate([state.flatten(self.use_idx) for state in states])
        actions = self.model(self.scaler.transform([states]))

        action = np.argmax(actions)
        rn = np.random.sample()

        if rn < 0.2 * eps:
            action = np.random.randint(len(self.step_sizes))
        elif rn < 0.6 * eps:
            action = min(action + 1, (len(self.step_sizes)) - 1)
        elif rn < eps:
            action = max(action - 1, 0)

        return self.step_sizes[action]

    def action_to_stepsize(self, action):
        """

        Parameters
        ----------
        action : int

        Returns
        -------
        float
            step_sizes[action]
        """
        return self.step_sizes[action]

    def get_actions(self, states):
        """
        Return the value of each possible action.

        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        np.ndarray
        """
        states = np.concatenate([state.flatten(self.use_idx) for state in states])
        return self.model(self.scaler.transform([states])).numpy()

    def train_on_batch(self, states, actions):
        """
        Train the model.

        Parameters
        ----------
        states : np.ndarray
        actions : np.ndarray
        """
        return self.model.train_on_batch(self.scaler.transform(states), actions)


class PredictorConstODE(PredictorODE):
    def __init__(self, h):
        """

        Parameters
        ----------
        h : float
            constant step size

        """
        self.h = h

    def __call__(self, states):
        """
        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        float
        """
        return self.h


class MetaQODE(PredictorODE):
    def __init__(self, basis_learners, model, scaler, use_idx=False):
        """
        Metalearner using a neural network model.

        Parameters
        ----------
        basis_learners : list[PredictorODE]
        model : tf.keras.Model
        scaler : StandardScaler
        use_idx : bool, optional
            whether the step size is used in the state or a idx referring to it
            (e.g. if the step sizes are from different orders like [0.0001, 0.0002, 0.1], idx might work better)
        """
        self.use_idx = use_idx
        self.basis_learners = basis_learners
        self.scaler = scaler
        self.model = model

    def __call__(self, states, eps=0):
        """
        Parameters
        ----------
        states : list[StateODE]
        eps : float
            probability that a random action is chosen instead of the one with highest value

        Returns
        -------
        float
            step_size
        """
        states_np = np.concatenate([state.flatten(self.use_idx) for state in states])
        actions = self.model(self.scaler.transform([states_np]))

        action = np.argmax(actions)
        rn = np.random.sample()

        if rn < 0.2 * eps:
            action = np.random.randint(len(self.basis_learners))
        elif rn < 0.6 * eps:
            action = min(action + 1, (len(self.basis_learners)) - 1)
        elif rn < eps:
            action = max(action - 1, 0)

        bl = self.basis_learners[action]
        return bl(states[:1])

    def action_to_stepsize(self, action, states):
        """

        Parameters
        ----------
        action : int
        states : list[StateODE]

        Returns
        -------
        float
            step_sizes[action]
        """
        bl = self.basis_learners[action]
        return bl(states[:1])

    def get_actions(self, states):
        """
        Return the value of each possible action.

        Parameters
        ----------
        states : list[StateODE]

        Returns
        -------
        np.ndarray
        """
        states = np.concatenate([state.flatten(self.use_idx) for state in states])
        return self.model(self.scaler.transform([states])).numpy()

    def train_on_batch(self, states, actions):
        """
        Train the model.

        Parameters
        ----------
        states : np.ndarray
        actions : np.ndarray
        """
        return self.model.train_on_batch(self.scaler.transform(states), actions)
