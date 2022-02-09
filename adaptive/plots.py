from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

__all__ = ["plot_trajectory"]


def plot_trajectory(
    x0: np.ndarray,
    timesteps: Sequence,
    nodes: Sequence,
    errors: Sequence,
    deltas: Sequence,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    error_tolerance: Optional[float] = 1e-5,
    figsize: Tuple[float, float] = (2.559, 2.402),
    linewidth: float = 0.5,
    marker: Optional[str] = None,
    axislabelsize: Union[str, int] = "x-small",
    ticklabelsize: Union[str, int] = "xx-small",
    axislinewidth: float = 0.5,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a trajectory.

    Parameters
    ----------
    x0 : ndarray
        Initial state.
    timesteps : sequence
        Sequence of timesteps.
    nodes : sequence
        Sequence of nodes.
    errors : sequence
        Sequence of errors achieved at each timestep.
    deltas : sequence
        Step sizes proposed by the neural network at each timestep.
    t_min : float, optional
        Minimum timestep to plot.
    t_max : float, optional
        Maximum timestep to plot.
    error_tolerance : float, optional
        Value of the horizontal line to plot in the error plot.
    figsize : tuple, optional (default: (2.559, 2.402))
        Figure size in inches. Defaults were calculated to fill half the page of the
        paper (6.5 x 6.1 cm) for a side by side comparison.
    linewidth : float, optional (default: 0.5)
        Line width of the trajectory, error and step size line plots.
    marker : str, optional (default: None)
        Marker to use for the plots. By default, no markers are used.
    axislabelsize : str or int, optional (default: "x-small")
        Font size of the axis labels.
    ticklabelsize : str or int, optional (default: "xx-small")
        Font size of the tick labels.
    axislinewidth : float, optional (default: 0.5)
        Line width of the axis spines, grid lines and tick lines.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plots.
    ax : matplotlib.axes.Axes
        Axes containing the plots.

    """
    if t_min is None:
        id_min = 0
    else:
        id_min = np.searchsorted(timesteps, t_min)
    if t_max is None:
        id_max = len(timesteps) - 1
    elif t_max < timesteps[-1]:
        id_max = np.searchsorted(timesteps, t_max, side="right") - 1
    else:
        id_max = len(timesteps) - 1

    times_to_plot = timesteps[id_min : id_max + 1]
    nodes_to_plot = np.array(nodes)[id_min : id_max + 1]

    errors = np.repeat(errors[id_min:id_max], 2)
    deltas = np.repeat(deltas[id_min:id_max], 2)

    times_duplicates = np.concatenate(
        ([times_to_plot[0]], np.repeat(times_to_plot[1:-1], 2), [times_to_plot[-1]])
    )

    dim = x0.shape[0]

    fig, ax = plt.subplots(dim + 2, figsize=figsize, sharex=True)

    # Plot the trajectory:
    color = "tab:blue"
    for i in range(dim):
        ax[i].plot(
            times_to_plot,
            nodes_to_plot[:, i],
            color=color,
            linewidth=linewidth,
            marker=marker,
        )
        ax[i].grid(linewidth=axislinewidth)
        ax[i].set_ylabel(f"$x_{i + 1}$", color=color, fontsize=axislabelsize)

    # Plot the errors:
    color = "tab:red"
    ax[dim].plot(
        times_duplicates, errors, color=color, linewidth=linewidth, marker=marker
    )
    ax[dim].set_ylabel("error", color=color, fontsize=axislabelsize)
    ax[dim].grid(linewidth=axislinewidth)
    ax[dim].axhline(error_tolerance, color="k", linewidth=linewidth)
    ax[dim].ticklabel_format(style="plain")
    ax[dim].yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: np.format_float_scientific(
                x, precision=2, trim="-", exp_digits=1
            )
        )
    )

    # Plot the stepsizes:
    color = "tab:blue"
    ax[dim + 1].plot(
        times_duplicates, deltas, color=color, linewidth=linewidth, marker=marker
    )
    ax[dim + 1].set_ylabel("step size", color=color, fontsize=axislabelsize)
    ax[dim + 1].grid(linewidth=axislinewidth)

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=ticklabelsize)
        a.tick_params(width=axislinewidth)
        plt.setp(a.spines.values(), linewidth=axislinewidth)

    fig.align_labels()

    # Adjust margins:
    left_in = 0.5
    right_in = 0.01
    bottom_in = 0.2
    top_in = 0.01
    hspace_in = 0.3
    plt.subplots_adjust(
        left=left_in / figsize[0],
        right=1.0 - right_in / figsize[0],
        bottom=bottom_in / figsize[1],
        top=1.0 - top_in / figsize[1],
        hspace=hspace_in / figsize[1],
    )

    return fig, ax
