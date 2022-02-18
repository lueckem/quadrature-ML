import numpy as np
from functions import Function, Sinus, SuperposeSinus
from matplotlib import pyplot as plt


class Romberg:
    def __init__(self, fun, a, b, tol=1e-8, order=10):
        """
        Parameters
        ----------
        fun : Function
        a : float
            start point
        b : float
            end point
        tol : float, optional
        order : int, optional
        """
        self.order = order  # k_max
        self.tol = tol
        self.b = b
        self.a = a
        self.f = fun

        self.n = [2 ** i for i in range(self.order)]
        self.A = [num + 1 for num in self.n]

        self.evals = 0
        self.nodes = []  # all nodes
        self.step_nodes = []  # nodes at start of every step
        self.orders = []  # order used for each step

    def reset(self):
        self.evals = 0
        self.nodes = []  # all nodes
        self.step_nodes = []  # nodes at start of every step
        self.orders = []  # order used for each step

    def __call__(self, h0, stepwise_errors=False):
        """
        Adaptive romberg quadrature with initial step size h0.

        If stepwise_errors=True it will also return an array of the errors of every step. (requires f to have a
        f.integral method)

        Parameters
        ----------
        h0 : float
            initial step size
        stepwise_errors : bool, optional
            if stepwise_errors=True it will also return an array of the errors of every step

        Returns
        -------
        integ_total : float
            integral
        errors : list[float], optional
            list of absolute error of every step
        """

        integ_total = 0.0
        errors = []
        a = self.a
        self.nodes.append(a)
        h = h0
        self.evals += 1
        while True:
            self.step_nodes.append(a)
            integ, next_h = self.one_step(a, h)

            if stepwise_errors:
                errors.append(abs(self.f.integral(a, a + h) - integ))

            self.evals -= 1  # the first node is counted twice
            integ_total += integ
            a = a + h
            h = min(next_h, self.b - a)
            if a >= self.b:
                break

        self.step_nodes.append(self.b)
        if stepwise_errors:
            return integ_total, errors
        return integ_total

    def one_step(self, a, step_size):
        b = a + step_size
        h = [(b - a) / num for num in self.n]
        f_vals = np.zeros(self.n[-1] + 1)
        f_vals[0] = self.f(a)
        f_vals[-1] = self.f(b)
        f_vals[int(self.n[-1] / 2)] = self.f(a + self.n[-1] / 2 * h[-1])
        # self.nodes.extend([a, b, a + self.n[-1] / 2 * h[-1]])
        self.nodes.extend([b, a + self.n[-1] / 2 * h[-1]])
        self.evals += 3
        this_order = 1

        # calculate tableau until error < tolerance or maximal order is reached
        tab = np.zeros((self.order, self.order))
        tab[0, 0] = h[0] * (f_vals[0] + f_vals[-1]) / 2
        for i in range(1, self.order):
            # fill i-th row of tableau
            delta = int(2 ** (self.order - 1) / (2 ** i))
            tab[i, 0] = np.sum(f_vals[::delta])
            tab[i, 0] -= (f_vals[0] + f_vals[-1]) / 2
            tab[i, 0] *= h[i]
            for j in range(1, i + 1):
                tab[i, j] = tab[i, j - 1] + (tab[i, j - 1] - tab[i - 1, j - 1]) / ((self.n[i] / self.n[i - j]) ** 2 - 1)
            this_order += 1

            # check error
            error = abs(tab[i, i - 1] - tab[i, i])
            if error < self.tol:
                break

            # calculate new f_vals
            if i + 1 < self.order:
                for j in range(int(delta / 2), len(f_vals), delta):
                    f_vals[j] = self.f(a + j * h[-1])
                    self.nodes.append(a + j * h[-1])
                    self.evals += 1
        tab = tab[:this_order, :this_order]
        self.orders.append(this_order)
        # print('order: {}'.format(this_order))

        # calculate optimal stepsizes
        errors = abs(np.diag(tab, -1) - np.diag(tab)[1:])
        h_opt = [(self.tol / e) ** (1.0 / (2 * idx + 3)) * step_size for idx, e in enumerate(errors)]
        work = [self.A[k + 1] / h_opt[k] for k in range(len(h_opt))]
        k_opt = np.argmin(work)
        h_opt = h_opt[k_opt]
        # print(h_opt)

        integ = tab[-1, -1]
        return integ, h_opt

    def get_nodes(self):
        return np.sort(self.nodes)

    def plot(self):
        self.nodes.sort()
        y_nodes = [self.f(node) for node in self.nodes]
        x = np.linspace(self.a, self.b, int((self.b - self.a) * 2000))
        y = [self.f(node) for node in x]

        # color = 'tab:blue'
        # plt.plot(x, y, color=color)
        # plt.xlabel('x', color='k')
        # plt.ylabel('f(x)', color=color)
        # plt.tick_params(axis='y', labelcolor=color)
        # color = 'tab:green'
        # plt.plot(self.nodes, y_nodes, 'x', color=color)
        # plt.plot(self.nodes, np.zeros(len(self.nodes)), '|', color=color)
        # plt.grid()

        fig, axs = plt.subplots(2, sharex=True, figsize=(10, 7), dpi=150, gridspec_kw={'height_ratios': [3, 1]})
        color = 'tab:blue'
        axs[0].set_xlabel('x', color='k')
        axs[0].set_ylabel('f(x)', color=color)
        axs[0].plot(x, y, color=color)
        axs[0].tick_params(axis='y', labelcolor=color)
        color = 'tab:green'
        axs[0].plot(self.nodes, y_nodes, 'x', color=color)
        axs[0].plot(self.step_nodes, np.zeros(len(self.step_nodes)), '|', color=color)
        axs[0].grid()

        color = 'tab:blue'
        axs[1].set_ylabel('order', color=color)
        orders = [val for val in self.orders for _ in (0, 1)]
        step_nodes = [self.step_nodes[0]] + [val for val in self.step_nodes[1:-1] for _ in (0, 1)] + [self.step_nodes[-1]]
        axs[1].plot(step_nodes, orders, 'x-', color=color)
        axs[1].grid()

        fig.tight_layout()
        plt.savefig('rom.png', dpi=150)
        plt.close()


def needleimpulse():
    def f(x):
        return 1 / (1e-4 + x ** 2)

    rom = Romberg(f, -1, 1, order=20, tol=1e-11)
    integ = rom(0.2)
    print(rom.step_nodes)
    print(len(rom.orders))
    print(integ)
    print('evaluations: {}'.format(rom.evals))
    print('global error: {}'.format(abs(integ - 312.159332)))
    # print(rom.nodes)
    rom.plot()


def test_sinus():
    f = Sinus()
    tol = 0.0005
    integ_rom = 0.0
    evals_rom_step = 0
    rom = Romberg(f, 0, 10, tol=tol, order=6)
    for j in range(10):
        rom = Romberg(f, 0, 10, tol=tol, order=6)
        integ_rom, errors = rom(0.15, True)
        error_rom_step = np.mean(errors)
        print(error_rom_step)
        evals_rom_step = rom.evals
        if error_rom_step < 0.0001:
            tol *= 2.0
        elif error_rom_step > 0.0005:
            tol /= 3.0
        else:
            break

    print('evaluations: {}'.format(evals_rom_step))
    print('global error: {}'.format(abs(integ_rom - f.integral(0, 20))))

    rom.plot()


if __name__ == '__main__':
    needleimpulse()
    # test_sinus()




