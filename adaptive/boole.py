import math
import numpy as np
from matplotlib import pyplot as plt


class BoolesRule:
    def __init__(self, f, a, b):
        """
        Parameters
        ----------
        f : Function
        a : float
            start point
        b : float
            end point
        """
        self.step_size = None
        self.num_steps = None
        self.b = b
        self.a = a
        self.f = f

        self.nodes = []

    def __call__(self, step_size=None, num_evals=None, stepwise_error=False):
        """
        Evaluate Integral using summed Booles rule.

        You can either specify step_size or the num_evals.

        Parameters
        ----------
        step_size : float, optional
        num_evals : float, optional
        stepwise_error : bool, optional
            If True returns (integ, error), where error is a list of absolute stepwise errors
        Returns
        -------
        integ : float
        errors : list[float], optional
        """

        if step_size is not None:
            self.num_steps = int(round((self.b - self.a) / step_size))
            self.num_steps = 4 * math.floor(self.num_steps / 4)  # self.num_steps needs to be multiple of 4
            self.step_size = (self.b - self.a) / self.num_steps
        elif num_evals is not None:
            self.num_steps = 4 * math.floor(num_evals / 4)  # self.num_steps needs to be multiple of 4
            self.step_size = (self.b - self.a) / self.num_steps
        else:
            raise ValueError('either step_size or num_evals hast to be specified.')

        integ = 0.0
        errors = []
        self.nodes = [self.a]
        for i in range(int(self.num_steps / 4)):
            a = self.nodes[-1]
            x2 = a + self.step_size
            x3 = x2 + self.step_size
            x4 = x3 + self.step_size
            b = x4 + self.step_size
            this_integ = boole(a, b, self.f(a), self.f(x2), self.f(x3), self.f(x4), self.f(b))
            integ += this_integ

            if stepwise_error:
                errors.append(abs(this_integ - self.f.integral(a, b)))

            self.nodes.extend([x2, x3, x4, b])

        if not stepwise_error:
            return integ
        else:
            return integ, errors

    @property
    def evals(self):
        return len(self.nodes)

    def plot(self):
        self.nodes.sort()
        y_nodes = [self.f(node) for node in self.nodes]
        x = np.linspace(self.a, self.b, int((self.b - self.a) * 20))
        y = [self.f(node) for node in x]

        color = 'tab:blue'
        plt.plot(x, y, color=color)
        plt.xlabel('x', color='k')
        plt.ylabel('f(x)', color=color)
        plt.tick_params(axis='y', labelcolor=color)
        color = 'tab:green'
        plt.plot(self.nodes, y_nodes, 'x', color=color)
        plt.plot(self.nodes, np.zeros(len(self.nodes)), '|', color=color)
        plt.grid()
        plt.savefig('booles.png', figsize=(10, 7), dpi=150)
        plt.close()


def boole(a, b, f1, f2, f3, f4, f5):
    return (b - a) / 90 * (7 * f1 + 32 * f2 + 12 * f3 + 32 * f4 + 7 * f5)
