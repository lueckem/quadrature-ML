from typing import Optional, Sequence, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

__all__ = ["plot_pareto", "plot_trajectory", "plot_trajectory_quad"]


def plot_pareto(
    model_data: np.ndarray,
    ode_data: np.ndarray,
    ode_norep_data: np.ndarray,
    opt_model_data: Optional[np.ndarray] = None,
    plot_scatter: bool = True,
    figsize: Tuple[float, float] = (2.559, 2.402),
    scatter_alpha: float = 0.4,
    axislabelsize: Union[str, int] = "x-small",
    ticklabelsize: Union[str, int] = "xx-small",
    legend_fontsize: Union[str, int] = "xx-small",
    modelmarkersize: float = 5,
    xrange: Optional[Tuple[float, float]] = None,
    yrange: Optional[Tuple[float, float]] = None,
    legend_ncol: int = 1,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the Pareto front of the model and the ODE data.

    Parameters
    ----------
    model_data : np.ndarray (2,) or (n_traj, 2)
        Array containing error and number of function evals. If the results for multiple
        trajectories are provided, a scatter plot will be plotted in addition to the
        mean point if plot_scatter is True.
    ode_data : np.ndarray (n_tol, 2)
        Array containing the errors and numbers of function evals for RK45 evaluated
        on different tolerances.
    ode_norep_data : np.ndarray (n_tol, 2)
        Same as ode_data, but rejections were not counted.
    opt_model_data : np.ndarray (2,) or (n_traj, 2)
        Array containing the error and number of function evals for the model with
        optimized weights. If the results for multiple trajectories are provided, a
        scatter plot will be plotted in addition to the mean point if plot_scatter is
        True.
    plot_scatter : bool
        If True, scatter plots will be plotted in addition to the mean points.
    figsize : Tuple[float, float]
        Size of the figure in inches.
    scatter_alpha : float (default: 0.1)
        Transparency of the points of the scatter plot.
    axislabelsize : Union[str, int]
        Size of the axis labels.
    ticklabelsize : Union[str, int]
        Size of the tick labels.
    legend_fontsize : Union[str, int]
        Size of the font in the legend.
    modelmarkersize : float (default: 5)
        Size of the markers depicting the mean point of the models.
    xrange : Tuple[float, float]
        Range of the x-axis. If None, the range will be determined automatically.
    yrange : Tuple[float, float]
        Range of the y-axis. If None, the range will be determined automatically.
    legend_ncol : int (default: 1)
        Number of columns in the legend.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot.
    ax : plt.Axes
        Axes object containing the plot.

    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the model data
    # If the model data contains more than one row, plot all rows as small scatter plot
    # and the mean a bit more pronounced:
    model_data = np.atleast_2d(model_data)
    if model_data.shape[0] > 1 and plot_scatter:
        ax.loglog(
            model_data[:, 0],
            model_data[:, 1],
            color="tab:red",
            marker="o",
            markeredgewidth=0,
            markersize=int(0.75 * modelmarkersize),
            linestyle="None",
            alpha=scatter_alpha,
        )
    mean = np.mean(model_data, axis=0)
    ax.loglog(
        mean[0],
        mean[1],
        color="tab:red",
        marker="o",
        markersize=modelmarkersize,
        linestyle="None",
        label="Model",
    )

    # Plot the optimized model data:
    if opt_model_data is not None:
        opt_model_data = np.atleast_2d(opt_model_data)
        if model_data.shape[0] > 1 and plot_scatter:
            ax.loglog(
                opt_model_data[:, 0],
                opt_model_data[:, 1],
                color="tab:purple",
                marker="H",
                markeredgewidth=0,
                markersize=int(0.75 * modelmarkersize),
                linestyle="None",
                alpha=scatter_alpha,
            )
        opt_mean = np.mean(opt_model_data, axis=0)
        ax.loglog(
            opt_mean[0],
            opt_mean[1],
            color="tab:purple",
            marker="H",
            markersize=modelmarkersize,
            linestyle="None",
            label="Model (optim. weights)",
        )

    # Plot the ODE data
    ax.loglog(
        ode_data[:, 0],
        ode_data[:, 1],
        color="tab:green",
        marker="x",
        label="RK45",
    )
    ax.loglog(
        ode_norep_data[:, 0],
        ode_norep_data[:, 1],
        color="tab:blue",
        marker="x",
        label="RK45 (rej. not counted)",
    )

    ax.legend(
        loc="best",
        fontsize=legend_fontsize,
        markerfirst=False,
        framealpha=0.6,
        ncol=legend_ncol,
        columnspacing=0,
    )
    ax.set_xlabel("error per RK step", fontsize=axislabelsize)
    ax.set_ylabel("$f$ evaluations per time", fontsize=axislabelsize)
    ax.grid(which="both")
    ax.tick_params(labelsize=ticklabelsize, which="both")
    plt.setp(ax.get_xminorticklabels(), visible=False)
    plt.setp(ax.get_yminorticklabels(), visible=False)

    if xrange is not None:
        ax.set_xlim(*xrange)
    if yrange is not None:
        ax.set_ylim(*yrange)

    # Adjust margins:
    left_in = 0.42
    right_in = 0.08
    bottom_in = 0.35
    top_in = 0.05
    plt.subplots_adjust(
        left=left_in / figsize[0],
        right=1.0 - right_in / figsize[0],
        bottom=bottom_in / figsize[1],
        top=1.0 - top_in / figsize[1],
    )

    return fig, ax


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
    color = "k"
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
    ax[dim + 1].set_ylabel("$h$", color=color, fontsize=axislabelsize)
    ax[dim + 1].grid(linewidth=axislinewidth)
    ax[dim + 1].set_xlabel("time", color="k", fontsize=axislabelsize)

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=ticklabelsize)
        a.tick_params(width=axislinewidth)
        plt.setp(a.spines.values(), linewidth=axislinewidth)

    fig.align_labels()

    # Adjust margins:
    left_in = 0.52
    right_in = 0.01
    bottom_in = 0.33
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


def plot_trajectory_quad(
    nodes: Sequence,
    f: Callable,
    errors: Sequence,
    deltas: Sequence,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
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
    nodes : sequence
        Sequence of nodes x.
    f : callable
        function f(x) that was integrated
    errors : sequence
        Sequence of errors achieved at each timestep.
    deltas : sequence
        Step sizes proposed by the neural network at each timestep.
    x_min : float, optional
        Minimum x to plot.
    x_max : float, optional
        Maximum x to plot.
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
    if x_min is None:
        id_min = 0
    else:
        id_min = np.searchsorted(nodes, x_min)

    if x_max is None:
        id_max = len(nodes) - 1
    elif x_max < nodes[-1]:
        id_max = np.searchsorted(nodes, x_max, side="right") - 1
    else:
        id_max = len(nodes) - 1

    nodes_to_plot = np.array(nodes)[id_min:id_max + 1]
    nodes_high_res = np.linspace(nodes_to_plot[0], nodes_to_plot[-1], len(nodes_to_plot) * 50)
    f_nodes = np.array([f(i) for i in nodes_high_res])

    errors = np.repeat(errors[id_min:id_max], 2)
    deltas = np.repeat(deltas[id_min:id_max], 2)

    nodes_duplicates = np.concatenate(
        ([nodes_to_plot[0]], np.repeat(nodes_to_plot[1:-1], 2), [nodes_to_plot[-1]])
    )

    fig, ax = plt.subplots(3, figsize=figsize, sharex=True)

    # Plot the trajectory:
    color = "k"
    ax[0].plot(
        nodes_high_res,
        f_nodes,
        color=color,
        linewidth=linewidth,
        marker=marker,
    )
    ax[0].grid(linewidth=axislinewidth)
    ax[0].set_ylabel("$f(t)$", color=color, fontsize=axislabelsize)

    # Plot the errors:
    color = "tab:red"
    ax[1].plot(
        nodes_duplicates, errors, color=color, linewidth=linewidth, marker=marker
    )
    ax[1].set_ylabel("error", color=color, fontsize=axislabelsize)
    ax[1].grid(linewidth=axislinewidth)
    ax[1].axhline(error_tolerance, color="k", linewidth=linewidth)
    ax[1].ticklabel_format(style="plain")
    ax[1].yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: np.format_float_scientific(
                x, precision=2, trim="-", exp_digits=1
            )
        )
    )

    # Plot the stepsizes:
    color = "tab:blue"
    ax[2].plot(
        nodes_duplicates, deltas, color=color, linewidth=linewidth, marker=marker
    )
    ax[2].set_ylabel("$h$", color=color, fontsize=axislabelsize)
    ax[2].grid(linewidth=axislinewidth)
    ax[2].set_xlabel("time", color="k", fontsize=axislabelsize)

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=ticklabelsize)
        a.tick_params(width=axislinewidth)
        plt.setp(a.spines.values(), linewidth=axislinewidth)

    fig.align_labels()

    # Adjust margins:
    left_in = 0.52
    right_in = 0.01
    bottom_in = 0.33
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
