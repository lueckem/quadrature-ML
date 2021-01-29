from matplotlib import pyplot as plt
import numpy as np
from functions import Function


class Simps:
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
        Evaluate Integral using summed Simpson rule.

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
            if self.num_steps % 2 == 1:
                self.num_steps -= 1
            self.step_size = (self.b - self.a) / self.num_steps
        elif num_evals is not None:
            if num_evals % 2 == 1:
                num_evals -= 1
            self.num_steps = num_evals
            self.step_size = (self.b - self.a) / self.num_steps
        else:
            raise ValueError('either step_size or num_evals hast to be specified.')

        integ = 0.0
        errors = []
        self.nodes = [self.a]
        for i in range(int(self.num_steps / 2)):
            a = self.nodes[-1]
            m = a + self.step_size
            b = m + self.step_size
            this_integ = simpson(a, b, self.f(a), self.f(m), self.f(b))
            integ += this_integ

            if stepwise_error:
                integ_true = self.f.integral(a, b)
                errors.append(abs(this_integ - integ_true))

            self.nodes.extend([m, b])

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
        plt.savefig('simps.png', figsize=(10, 7), dpi=150)
        plt.close()


class AdaptSimps:
    def __init__(self, f, a, b):
        self.b = b
        self.a = a
        self.f = f

        self.nodes = []

    def __call__(self, eps):
        fa, fb = self.f(self.a), self.f(self.b)
        self.nodes.append(self.a)
        self.nodes.append(self.b)
        m, fm, whole = self._quad_simpsons_mem(self.a, fa, self.b, fb)
        return self._quad_asr(self.a, fa, self.b, fb, eps, whole, m, fm)

    def _quad_simpsons_mem(self, a, fa, b, fb):
        """Evaluates the Simpson's Rule, also returning m and f(m) to reuse"""
        m = (a + b) / 2
        fm = self.f(m)
        self.nodes.append(m)
        return m, fm, abs(b - a) / 6 * (fa + 4 * fm + fb)

    def _quad_asr(self, a, fa, b, fb, eps, whole, m, fm):
        """
        Efficient recursive implementation of adaptive Simpson's rule.
        Function values at the start, middle, end of the intervals are retained.
        """
        lm, flm, left = self._quad_simpsons_mem(a, fa, m, fm)
        rm, frm, right = self._quad_simpsons_mem(m, fm, b, fb)
        delta = left + right - whole
        if abs(delta) <= 15 * eps:
            return left + right + delta / 15
        return self._quad_asr(a, fa, m, fm, eps / 2, left, lm, flm) + self._quad_asr(m, fm, b, fb, eps / 2, right, rm,
                                                                                     frm)

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
        plt.savefig('adapt_simps.png', figsize=(10, 7), dpi=150)
        plt.close()


class AdaptSimpsConstEvals:
    """
    Adaptive Simpson rule, but the splitting of intervals is not done by some threshold, but until a specified
        number of function evaluations is reached
    """

    def __init__(self, f, a, b):
        self.b = b
        self.a = a
        self.f = f
        self.evals = 0
        self.intervals = []

    def __call__(self, max_evals):
        # set first interval
        nodes = [self.a, 0.75 * self.a + 0.25 * self.b, (self.a + self.b) / 2, 0.25 * self.a + 0.75 * self.b, self.b]
        f_nodes = [self.f(node) for node in nodes]
        self.intervals.append(Interval(nodes, f_nodes))
        self.evals += 5

        # split intervals until max_evals are reached
        while self.evals < max_evals:
            # find interval with highest error
            max_idx = max(range(len(self.intervals)), key=lambda i: self.intervals[i].error)
            max_interval = self.intervals.pop(max_idx)
            new1, new2 = self.split_interval(max_interval)
            self.intervals.extend((new1, new2))

        return self.integral

    def split_interval(self, interval):
        m11 = (interval.a + interval.m1) / 2
        m13 = (interval.m1 + interval.m2) / 2
        m21 = (interval.m2 + interval.m3) / 2
        m23 = (interval.m3 + interval.b) / 2

        nodes1 = [interval.a, m11, interval.m1, m13, interval.m2]
        nodes2 = [interval.m2, m21, interval.m3, m23, interval.b]
        f_nodes1 = [interval.fa, self.f(m11), interval.fm1, self.f(m13), interval.fm2]
        f_nodes2 = [interval.fm2, self.f(m21), interval.fm3, self.f(m23), interval.fb]
        self.evals += 4

        new1 = Interval(nodes1, f_nodes1)
        new2 = Interval(nodes2, f_nodes2)
        return new1, new2

    def plot(self):
        nodes = self.nodes
        y_nodes = [self.f(node) for node in nodes]
        x = np.linspace(self.a, self.b, int((self.b - self.a) * 20))
        y = [self.f(node) for node in x]

        color = 'tab:blue'
        plt.plot(x, y, color=color)
        plt.xlabel('x', color='k')
        plt.ylabel('f(x)', color=color)
        plt.tick_params(axis='y', labelcolor=color)
        color = 'tab:green'
        plt.plot(nodes, y_nodes, 'x', color=color)
        plt.plot(nodes, np.zeros(len(nodes)), '|', color=color)
        plt.grid()
        plt.savefig('adapt_simps_const.png', figsize=(10, 7), dpi=150)
        plt.close()

    @property
    def integral(self):
        integ = 0.0
        for interval in self.intervals:
            integ += interval.integral
        return integ

    @property
    def nodes(self):
        nodes = []
        for interval in self.intervals:
            nodes.extend(interval.nodes)
        nodes = list(set(nodes))
        nodes.sort()
        return nodes

    @property
    def stepwise_errors(self):
        """ self.f needs to have f.integral method! """
        errors = []
        for interval in self.intervals:
            errors.append(abs(interval.I1 - self.f.integral(interval.a, interval.m2)))
            errors.append(abs(interval.I2 - self.f.integral(interval.m2, interval.b)))
        return errors


class Interval:
    def __init__(self, nodes, f_nodes):
        """ nodes = (a, m1, m2, m3, b)
            f_nodes = (f(a), ..., f(b)) """
        self.a, self.m1, self.m2, self.m3, self.b = nodes
        self.fa, self.fm1, self.fm2, self.fm3, self.fb = f_nodes

        self.I_whole = simpson(self.a, self.b, self.fa, self.fm2, self.fb)
        self.I1 = simpson(self.a, self.m2, self.fa, self.fm1, self.fm2)
        self.I2 = simpson(self.m2, self.b, self.fm2, self.fm3, self.fb)

        self.error = abs(self.I1 + self.I2 - self.I_whole)

    @property
    def integral(self):
        # delta = self.I1 + self.I2 - self.I_whole
        # return self.I1 + self.I2 + delta / 15
        return self.I1 + self.I2

    @property
    def nodes(self):
        return [self.a, self.m1, self.m2, self.m3, self.b]


def simpson(a, b, fa, fm, fb):
    return abs(b - a) / 6 * (fa + 4 * fm + fb)


if __name__ == '__main__':
    # test ASR
    import scipy.integrate as si

    def f(x):
        if x < 10:
            return np.sin(x)
        return np.sin(5 * x)

    a = 0
    b = 13
    integ_correct, tol, info = si.quad(f, a, b, full_output=1)
    print(info['neval'])
    asr = AdaptSimpsConstEvals(f, a, b)
    integ = asr(200)
    print(integ, integ_correct)
    asr.plot()
