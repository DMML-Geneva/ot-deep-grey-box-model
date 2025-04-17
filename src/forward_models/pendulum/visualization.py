from matplotlib import pyplot as plt
import numpy as np


def plot_phase_space(v, dir_path=".", file_name="phase_space.png"):
    fig = plt.figure(figsize=(10, 6))
    ax0 = fig.add_subplot(111)

    ax0.plot([p[0] for p in v], [p[1] for p in v])

    ax0.set_xlabel(r"$x$", fontsize=25)
    ax0.set_ylabel(r"$\sin(x)$", fontsize=25)
    ax0.set_title("Phase time", fontsize=25)
    ax0.set_aspect("equal")
    plt.show()
    plt.savefig(f"{dir_path}/{file_name}")


def plot_trajectory(
    theta, ts, savefig=False, dir_path=".", file_name="trajectory.png"
):
    fig = plt.figure(figsize=(10, 6))
    ax0 = fig.add_subplot(111)

    ax0.plot(ts, theta)

    ax0.set_xlabel(r"$t$", fontsize=25)
    ax0.set_ylabel(r"$x$", fontsize=25)
    ax0.set_title("Trajectory", fontsize=25)
    ax0.set_aspect("equal")
    plt.show()
    if savefig:
        plt.savefig(f"{dir_path}/{file_name}")


def plot_traj(
    t,
    test_params,
    test_sims,
    pred,
    X_sims,
    X_init_conds=None,
    n_plot_samples=10,
    rnd_samples=True,
    save_path=None,
):
    # Predict the trajectories
    # Plot the predicted trajectories

    for i in range(n_plot_samples):
        if rnd_samples:
            idx = np.random.randint(0, len(test_params))
        else:
            idx = i
        idx = np.random.randint(0, len(test_params))
        print(f"Sample {idx}")

        fig = plt.figure(figsize=(6, 6))
        ax0 = fig.add_subplot(111)

        # incomplete trajectories
        if X_sims is not None:
            ax0.plot(
                t, X_sims[idx].tolist(), color="gray", label="Part", alpha=0.4
            )

        # complete trajectories
        z_size = test_sims.shape[1] if test_sims.ndim > 2 else 1
        for j in range(z_size):
            sims = test_sims[idx][j] if z_size > 1 else test_sims[idx]
            ax0.plot(
                t,
                sims.flatten().tolist(),
                color="blue",
                label="Full",
                alpha=0.4,
            )

        # predicted params
        z_size = pred.shape[1] if pred.ndim > 2 else 1
        for j in range(z_size):
            p = pred[idx][j] if z_size > 1 else pred[idx]
            ax0.plot(
                t,
                p.flatten().tolist(),
                color="orange",
                label="Pred",
                alpha=0.6,
            )

        # init_cond = X_init_conds[idx][0].item()
        if test_params[idx].shape[-1] > 1:
            param_1 = test_params[idx][0][0].item()
            if test_params[idx][0].shape[0] > 1:
                param_2 = test_params[idx][0][1].item()
            if test_params[idx][0].shape[0] > 2:
                param_3 = test_params[idx][0][2].item()
            else:
                param_3 = 0
            if test_params[idx][0].shape[0] > 3:
                param_4 = test_params[idx][0][3].item()
            else:
                param_4 = 0

            title = r"Full: $\omega$={:.2f} $\xi$={:.2f}, A={:.2f}, $\phi$={:.2f}".format(
                param_1, param_2, param_3, param_4
            )

        else:
            if test_params.shape[0] > 1:
                param_1 = test_params[idx][0].item()
            else:
                param_1 = test_params[idx].item()
            title = r"Full: $\omega$={:.2f}".format(param_1)

        handles, labels = ax0.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Angle")
        ax0.set_title(title)
        if save_path is not None:
            plt.savefig(save_path + f"/{i}.png")
        else:
            plt.show()

    return fig


def plot_trajectories(
    params,
    sims,
    init_conds,
    ts,
    savefig=False,
    dir_path=".",
    file_name="trajectory.pdf",
    show_legend=True,
    alpha=1,
    plot_show=True,
    figsize=(12, 12),
    ylim=[-1.57, 1.57],
    only_color_legend=False,
):
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(111)

    legends = []
    title = None
    for i in range(len(params)):
        if isinstance(params[i], float) or len(params[i]) == 1:
            init_cond = (
                init_conds[i]
                if isinstance(init_conds[i], float)
                else init_conds[i][0]
            )
            param_2 = (
                params[i] if isinstance(params[i], float) else params[i][0]
            )
            # add legend for sims param_1 and param_2
            color = None
            if show_legend:
                if isinstance(param_2, str):
                    str_legend = r"Pred"
                    if not only_color_legend:
                        str_legend = r"Pred: {}".format(param_2)
                    color = "blue"
                else:
                    if not only_color_legend:
                        str_legend = r"Part: $x_0$={:.2f}, $\omega$={:.2f}".format(
                            init_cond, param_2
                        )
                    else:
                        str_legend = r"Part"
                        color = "gray"
                if str_legend not in legends:
                    legends.append(str_legend)
                ax0.plot(ts, sims[i], alpha=alpha)
            else:
                ax0.plot(ts, sims[i], alpha=alpha)
        else:
            init_cond = init_conds[i][0]
            # add legend for thetas param_1 and param_2
            if show_legend:
                color = None
                if isinstance(params[i], str):
                    if not only_color_legend:
                        str_legend = r"Pred: {}".format(params[i])
                    else:
                        str_legend = r"Pred"
                        color = "blue"
                else:
                    param_2, param_3, param_4, param_5 = params[i]
                    if not only_color_legend:

                        str_legend = r"Full: $x_0$={:.2f}, $\omega$={:.2f}, $\xi$={:.2f}, A={:.2f}, $\phi$={:.2f}".format(
                            init_cond, param_2, param_3, param_4, param_5
                        )
                    else:
                        title = r"Full: $x_0$={:.2f}, $\omega$={:.2f}, $\xi$={:.2f}, A={:.2f}, $\phi$={:.2f}".format(
                            init_cond, param_2, param_3, param_4, param_5
                        )
                        str_legend = "Full"
                        color = "gray"
                if str_legend not in legends:
                    legends.append(str_legend)

                if color == None:
                    ax0.plot(ts, sims[i], alpha=alpha)
                else:
                    ax0.plot(ts, sims[i], alpha=alpha, color=color)
            else:
                ax0.plot(ts, sims[i], alpha=alpha)
        if show_legend:
            ax0.legend(legends, loc="upper right")
        elif len(params[i]) != 1:
            init_cond = init_conds[0][0]
            # param_2 = params[i][:1].item()
            # ax0.set_title(
            #    r"Trajectory param=[{:.2f}, {:.2f}]".format(init_cond, param_2)
            # )
    if ylim is not None:
        ax0.set_ylim(ylim)

    ax0.set_xlabel(
        r"$t$",
    )
    ax0.set_ylabel(
        r"$x$",
    )
    if title is not None:
        ax0.set_title(title)

    #ax0.set_aspect("equal")
    plt.margins(0,0)
    #fig.tight_layout()
    if savefig:
        plt.savefig(f"{dir_path}/{file_name}")
        print(f"Saved figure to {dir_path}/{file_name}")
        
    if plot_show:
        plt.show()
    
    return fig, ax0

def plot_trajectories_minimal(
    params,
    sims,
    init_conds,
    ts,
    savefig=False,
    dir_path=".",
    file_name="trajectory.pdf",
    show_legend=True,
    alpha=1,
    plot_show=True,
    figsize=(8, 5),  # Adjusted default figsize for better presentation
    ylim=[-1.57, 1.57],
    only_color_legend=False,
    cmap_theme='viridis',  # Added colormap parameter
):
    """
    Plots trajectories with a minimal style (no solid axes, just tick marks)
    suitable for ML conferences.

    Args:
        params (list): List of parameters for each simulation.
        sims (list): List of simulation results (trajectories).
        init_conds (list): List of initial conditions for each simulation.
        ts (np.ndarray): Time array.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        dir_path (str, optional): Directory to save the figure to. Defaults to ".".
        file_name (str, optional): Name of the saved figure file. Defaults to "trajectory.pdf".
        show_legend (bool, optional): Whether to show the legend. Defaults to True.
        alpha (float, optional): Transparency of the plotted lines. Defaults to 1.
        plot_show (bool, optional): Whether to display the plot. Defaults to True.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 5).
        ylim (list, optional): Limits for the y-axis. Defaults to [-1.57, 1.57].
        only_color_legend (bool, optional): Whether to show only a color-based legend. Defaults to False.
    """
    plt.style.use('seaborn-v0_8-white')  # Use a modern style without grid
    fig, ax = plt.subplots(figsize=figsize)
    colors=None if cmap_theme is None else plt.cm.get_cmap(cmap_theme, len(params))
    legends = []
    title = None

    for i, param in enumerate(params):
        init_cond = init_conds[i] if isinstance(init_conds[i], (float, int)) else init_conds[i][0]
        sim = sims[i]
        color = None if colors is None else colors(i)

        if isinstance(param, (float, int)) or len(param) == 1:
            param_val = param if isinstance(param, (float, int)) else param[0]
            if show_legend:
                if isinstance(param_val, str):
                    label = "Prediction" if only_color_legend else f"Prediction: {param_val}"
                else:
                    label = "Partial" if only_color_legend else rf"Partial: $x_0$={init_cond:.2f}, $\omega$={param_val:.2f}"
                if label not in legends:
                    legends.append(label)
                if color is not None:
                    ax.plot(ts, sim, alpha=alpha, label=label, color=color)
                else:  
                    ax.plot(ts, sim, alpha=alpha, label=label)
            else:
                if color is not None:
                    ax.plot(ts, sim, alpha=alpha, color=color)
                else:
                    ax.plot(ts, sim, alpha=alpha)
        else:
            if show_legend:
                if isinstance(param, str):
                    label = "Prediction" if only_color_legend else f"Prediction: {param}"
                    ax.plot(ts, sim, alpha=alpha, label=label, color='blue') # Explicit color for string params
                    if label not in legends:
                        legends.append(label)
                else:
                    param_2, param_3, param_4, param_5 = param
                    if not only_color_legend:
                        label = rf"Full: $x_0$={init_cond:.2f}, $\omega$={param_2:.2f}, $\xi$={param_3:.2f}, A={param_4:.2f}, $\phi$={param_5:.2f}"
                    else:
                        title_str = rf"Full: $x_0$={init_cond:.2f}, $\omega$={param_2:.2f}, $\xi$={param_3:.2f}, A={param_4:.2f}, $\phi$={param_5:.2f}"
                        ax.set_title(title_str)
                        label = "Full"
                    if label not in legends:
                        legends.append(label)
                    if color is not None:
                        ax.plot(ts, sim, alpha=alpha, label=label, color=color)
                    else:  
                        ax.plot(ts, sim, alpha=alpha, label=label)
            else:
                if color is not None:
                    ax.plot(ts, sim, alpha=alpha, color=color)
                else:
                    ax.plot(ts, sim, alpha=alpha)

    if show_legend:
        ax.legend(loc="upper right", frameon=True) # Add a frame to the legend

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")

    if title is not None:
        ax.set_title(title)

    # Remove all spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Show ticks only
    ax.tick_params(axis='both', which='both', length=5) # Adjust tick length
    ax.tick_params(axis='both', which='major', labelsize=10) # Improve tick label size

    fig.tight_layout() # Adjust layout to prevent overlapping elements
    plt.margins(0, 0) # Remove x margin

    if savefig:
        plt.savefig(f"{dir_path}/{file_name}", bbox_inches='tight') # Use bbox_inches for cleaner saving
        print(f"Saved figure to {dir_path}/{file_name}")

    if plot_show:
        plt.show()


def plot_simulated_trajectories(
    test_params, test_sims, t, idx_start=0, n_samples=9, rows=3, cols=3
):
    params = test_params[idx_start : n_samples + idx_start]
    simulations = test_sims[idx_start : n_samples + idx_start]
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    theta_1, theta_2 = params[0, :2]

    for i in range(rows):
        for j in range(cols):
            axes[i, j].plot(
                t,
                simulations[i * rows + j],
                linestyle="-",
                marker="o",
                markersize=1.4,
            )
            axes[i, j].set_title(
                r"$\xi$={params[i*rows+j, 2]:.2f}, A={params[i*rows+j, 3]:.2f}, $\phi$={params[i*rows+j, 3]:.2f}"
            )
            axes[i, j].set_aspect("equal")
            axes[i, j].set_xlabel(r"$t$")
            axes[i, j].set_ylabel(r"$x$")
    # set global title to the figure
    fig.suptitle(
        r"Simulated angle trajectories params=[{:.2f}, {:.2f}]".format(
            theta_1, theta_2
        )
    )
    # add spacing between subplots
    fig.tight_layout()
    plt.show()


def subplot_trajectories(
    test_params,
    test_sims,
    t,
    idx_start=0,
    n_samples=9,
    rows=3,
    cols=3,
    show_legend=True,
    alpha=1,
    figsize=(12, 12),
    sharey=True,
    wspace=None,
    hspace=None,
    tight_layout=True,
    plot_show=True,
    ylim=[-1.57, 1.57],
    aspect=None,
    xticks=None,
):
    params = test_params[idx_start:]
    simulations = test_sims[idx_start:]
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, sharex=True, sharey=sharey
    )
    axes = np.atleast_2d(axes)
    if cols == 1:
        axes = axes.T

    if wspace is not None and hspace is not None:
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

    idx_sims = 0
    for i in range(rows):
        for j in range(cols):
            ll = []
            for k in range(n_samples):
                # set legend
                item = params[idx_sims + k]
                if type(item) == str:
                    str_legend = r"Pred"
                else:
                    if len(item) > 2:
                        str_legend = r"$Full: \xi$={:.2f}, A={:.2f}, $\phi$={:.2f}".format(
                            item[1], item[2], item[3]
                        )
                    else:
                        str_legend = r"Part: $x_0$={:.2f}, $\omega$={:.2f}".format(
                            simulations[idx_sims][0], item[0]
                        )

                if str_legend not in ll:
                    ll.append(str_legend)
                    sims_to_plot = simulations[idx_sims + k]
                    length = (
                        len(sims_to_plot)
                        if isinstance(sims_to_plot[0], list)
                        else 1
                    )
                    for idx in range(length):
                        axes[i, j].plot(
                            t,
                            sims_to_plot[idx] if length > 1 else sims_to_plot,
                            alpha=alpha,
                            linestyle="-",
                            marker="o",
                            markersize=1.4,
                        )
                        axes[i, j].set_ylim(ylim)

            if show_legend:
                axes[i, j].legend(ll)
            axes[i, j].set_title(
                r"[$x_0$={:.2f}, $\omega$={:.2f}]".format(
                    simulations[idx_sims][0], params[idx_sims][0]
                )
            )
            axes[i, j].set_aspect("equal")
            axes[i, j].set_xlabel(r"$t$")
            axes[i, j].set_ylabel(r"$x$")
            if aspect is None:
                axes[i, j].set_aspect("equal")
            else:
                axes[i, j].set_aspect(aspect)
            # xticks
            if xticks is not None:
                axes[i, j].set_xticks(xticks)

            idx_sims += n_samples

    # add spacing between subplots
    if tight_layout:
        fig.tight_layout()
    if plot_show:
        plt.show()
    return fig, axes
