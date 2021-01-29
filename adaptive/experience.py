import numpy as np
from adaptive.integrator import StateODE


class Experience:
    def __init__(self, batch_size):
        """
        Parameters
        ----------
        batch_size : int
            Number of experiences returned in the get_samples method.
        """
        self.batch_size = batch_size
        self.current = []

    def reset(self):
        """
        Delete stored ecperiences.
        """
        self.current = []

    def append(self, state, target):
        """
        Parameters
        ----------
        state : np.ndarray
            shape (dim_state,)
        target : np.ndarray
            shape (dim_action,)
        """
        self.current.append((state, target))

    def is_full(self):
        return len(self.current) >= self.batch_size

    def get_samples(self):
        # get first half from current experience (if possible)
        out = self.current[-self.batch_size:]
        out = [item for item in zip(*out)]
        return np.stack(out[0]), np.stack(out[1])


class ExperienceODE(Experience):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def append(self, state, target):
        """
        Parameters
        ----------
        state : list[StateODE]
        target : np.ndarray
            shape (dim_action,)
        """
        self.current.append((state, target))

    def get_samples(self, use_idx=False):
        out = self.current[-self.batch_size:]

        out = [item for item in zip(*out)]
        out[0] = [np.concatenate([state.flatten(use_idx) for state in out[0][i]]) for i in range(len(out[0]))]
        return np.stack(out[0]), np.stack(out[1])
