"""Plots for the blob-curriculum study (load a result dict, return a Figure)."""

import numpy as np
import matplotlib.pyplot as plt

_COND_COLOR = {"standard": "C0", "curriculum": "C1", "anti_curriculum": "C2"}
_COND_LABEL = {"standard": "standard", "curriculum": "curriculum",
               "anti_curriculum": "anti-curriculum"}
CONDITION_ORDER = ("standard", "curriculum", "anti_curriculum")


def _conditions_present(main, key):
    """Conditions (in canonical order) that have a ``<cond>_<key>`` array."""
    return [c for c in CONDITION_ORDER if f"{c}_{key}" in main]


def _nearest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def _descriptor(main):
    """Short run descriptor for suptitles (width, seeds, init scale, acc mode)."""
    d = f"width = {main['n_hidden']} hidden units, {main['num_seeds']} seeds"
    if "weight_init_std" in main:
        d += f", init scale {main['weight_init_std']}"
    if main.get("accuracy_mode") == "expected":
        d += ", expected acc"
    return d


def _acc_prefix(main):
    """'expected ' when the accuracy metric is the soft/sigmoid one, else ''."""
    return "expected " if main.get("accuracy_mode") == "expected" else ""


def _line_with_band(ax, x, main, cond, key, i, label=None):
    """Plot a condition's curve for noise-index ``i`` with a +/-1 SEM band over
    seeds (shaded) if a matching ``<cond>_<key>_sem`` array is present."""
    mean = main[f"{cond}_{key}"][i]
    ax.plot(x, mean, "-o", ms=3, color=_COND_COLOR[cond],
            label=label or _COND_LABEL.get(cond, cond))
    sem = main.get(f"{cond}_{key}_sem")
    if sem is not None:
        ax.fill_between(x, mean - sem[i], mean + sem[i],
                        color=_COND_COLOR[cond], alpha=0.2, lw=0)


def plot_benefit_vs_noise(main):
    """Final accuracy and curriculum benefit vs input noise std (3 panels).

    Left: final exception (peripheral) accuracy per condition. Middle: benefit
    over standard on the exceptions (curriculum - std, and anti-curriculum - std
    if present). Right: final rule (central) accuracy per condition."""
    noise = main["noise_levels"]
    xlabel = "input noise std (signal_noise_std)"
    pre = _acc_prefix(main)
    std_p = main["standard_peripheral"][:, -1]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for cond in _conditions_present(main, "peripheral"):
        ax[0].plot(noise, main[f"{cond}_peripheral"][:, -1], "-o",
                   color=_COND_COLOR[cond], label=_COND_LABEL[cond])
        ax[2].plot(noise, main[f"{cond}_central"][:, -1], "-o",
                   color=_COND_COLOR[cond], label=_COND_LABEL[cond])
    ax[0].axhline(0.5, ls=":", c="gray")
    ax[0].set(xlabel=xlabel, ylabel=f"final {pre}exception accuracy", ylim=(-0.02, 1.02),
              title=f"Final {pre}exception accuracy vs noise")
    ax[0].legend()
    # benefit over standard, for each non-standard condition
    for cond in _conditions_present(main, "peripheral"):
        if cond == "standard":
            continue
        ax[1].plot(noise, main[f"{cond}_peripheral"][:, -1] - std_p, "-o",
                   color=_COND_COLOR[cond], label=f"{_COND_LABEL[cond]} - std")
    ax[1].axhline(0.0, ls=":", c="gray")
    ax[1].set(xlabel=xlabel, ylabel=f"benefit over standard on {pre}exceptions",
              title="Benefit vs noise")
    ax[1].legend()
    ax[2].axhline(0.5, ls=":", c="gray")
    ax[2].set(xlabel=xlabel, ylabel=f"final {pre}rule accuracy", ylim=(-0.02, 1.02),
              title=f"Final {pre}rule accuracy vs noise")
    ax[2].legend()
    fig.suptitle(f"{_descriptor(main)}, {main['checkpoints'][-1]} steps", y=1.02)
    fig.tight_layout()
    return fig


_GROUP_LABEL = {"peripheral": "exception", "central": "rule"}


def plot_noise_x_steps(main, group="peripheral"):
    """Heatmap of curriculum benefit over (input noise x training length).

    ``group`` = 'peripheral' (exceptions, default) or 'central' (rule). For the
    rule, a blue region means curriculum *hurts* rule accuracy (the network
    trades rule for exception performance)."""
    noise, steps = main["noise_levels"], main["checkpoints"]
    benefit = main[f"curriculum_{group}"] - main[f"standard_{group}"]   # (K, T)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(benefit, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                   aspect="auto", origin="lower",
                   extent=[steps[0], steps[-1], noise[0], noise[-1]])
    ax.set(xlabel="training steps", ylabel="input noise std",
           title=f"Curriculum benefit on {_GROUP_LABEL[group]} accuracy "
                 f"(curr - std), width={main['n_hidden']}")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    return fig


def _accuracy_curve_grid(main, group, ylabel, title, noise_targets=None):
    """Accuracy learning curves (std vs curr) at representative noise levels for
    one stimulus group ('peripheral' = exceptions, 'central' = rule)."""
    noise = main["noise_levels"]
    reps = main["representative_noise"] if noise_targets is None else noise_targets
    steps = main["checkpoints"]
    fig, ax = plt.subplots(1, len(reps), figsize=(4.4 * len(reps), 4), sharey=True)
    ax = np.atleast_1d(ax)
    conds = _conditions_present(main, group)
    for k, target in enumerate(reps):
        i = _nearest_index(noise, target)
        for cond in conds:
            _line_with_band(ax[k], steps, main, cond, group, i)
        ax[k].axhline(0.5, ls=":", c="gray")
        ax[k].set(title=f"noise std = {noise[i]:.2f}", xlabel="training step",
                  ylim=(-0.02, 1.02))
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel(_acc_prefix(main) + ylabel)
    fig.suptitle(f"{title} ({_descriptor(main)})", y=1.02)
    fig.tight_layout()
    return fig


def plot_learning_curves(main, noise_targets=None):
    """Exception (peripheral) accuracy learning curves at representative noise.

    ``noise_targets`` overrides the stored representative noise levels (snapped
    to the nearest available level)."""
    return _accuracy_curve_grid(main, "peripheral", "exception accuracy",
                                "Exception learning curves vs input noise",
                                noise_targets)


def plot_rule_learning_curves(main, noise_targets=None):
    """Rule (central) accuracy learning curves at representative noise levels."""
    return _accuracy_curve_grid(main, "central", "rule accuracy",
                                "Rule learning curves vs input noise",
                                noise_targets)


def _metric_curve_grid(main, key, ylabel, title, noise_targets=None,
                       ylim=None, hline=None):
    """Generic metric-over-training grid (std vs curr, SEM band) across
    representative noise levels -- used by the mechanism plots."""
    noise = main["noise_levels"]
    reps = main["representative_noise"] if noise_targets is None else noise_targets
    steps = main["checkpoints"]
    fig, ax = plt.subplots(1, len(reps), figsize=(4.4 * len(reps), 4), sharey=True)
    ax = np.atleast_1d(ax)
    conds = _conditions_present(main, key)
    for k, target in enumerate(reps):
        i = _nearest_index(noise, target)
        for cond in conds:
            _line_with_band(ax[k], steps, main, cond, key, i)
        if hline is not None:
            ax[k].axhline(hline, ls=":", c="gray")
        if ylim is not None:
            ax[k].set_ylim(*ylim)
        ax[k].set(title=f"noise std = {noise[i]:.2f}", xlabel="training step")
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel(ylabel)
    fig.suptitle(f"{title} ({_descriptor(main)})", y=1.02)
    fig.tight_layout()
    return fig


def plot_participation_ratio(main, noise_targets=None):
    """Participation ratio (effective dimensionality) of the 8 centroids'
    post-ReLU hidden representation over training. Higher = less compressed;
    the uncompression hypothesis predicts curriculum stays above standard."""
    return _metric_curve_grid(
        main, "participation_ratio",
        "participation ratio (8 centroids, post-ReLU)",
        "Hidden-representation dimensionality over training",
        noise_targets, ylim=(0.8, 5.0))


def plot_active_units(main, noise_targets=None):
    """Fraction of hidden units firing on >= 1 centroid over training
    (unit recruitment -- the mechanistic partner to participation ratio)."""
    return _metric_curve_grid(
        main, "active_units", "fraction of hidden units active",
        "Active hidden units over training", noise_targets, ylim=(-0.02, 1.02))


def plot_loss_curves(main, noise_targets=None):
    """BCE loss over training at representative noise levels: rows are rule
    (central) loss, exception (peripheral) loss, and combined (all 8); columns
    are noise levels; standard vs curriculum overlaid. Loss is on the clean
    prototypes (the blob centres)."""
    noise = main["noise_levels"]
    reps = main["representative_noise"] if noise_targets is None else noise_targets
    steps = main["checkpoints"]
    rows = [("central_loss", "rule loss (6 central)"),
            ("peripheral_loss", "exception loss (2 peripheral)"),
            ("overall_loss", "combined loss (all 8)")]
    # share y within each row (metric) so noise columns are directly comparable
    fig, ax = plt.subplots(len(rows), len(reps),
                           figsize=(4.4 * len(reps), 3.2 * len(rows)),
                           sharex=True, sharey="row")
    ax = np.array(ax).reshape(len(rows), len(reps))
    for r, (metric, ylabel) in enumerate(rows):
        conds = _conditions_present(main, metric)
        for k, target in enumerate(reps):
            i = _nearest_index(noise, target)
            for cond in conds:
                _line_with_band(ax[r, k], steps, main, cond, metric, i)
            if r == 0:
                ax[r, k].set_title(f"noise std = {noise[i]:.2f}")
            if r == len(rows) - 1:
                ax[r, k].set_xlabel("training step")
            if k == 0:
                ax[r, k].set_ylabel(ylabel)
            ax[r, k].legend(fontsize=8)
    fig.suptitle(f"BCE loss on the blob centres: rule vs exception vs combined "
                 f"({_descriptor(main)})", y=1.01)
    fig.tight_layout()
    return fig


def plot_noise_x_hidden(hidden_sweep):
    """Heatmap of final curriculum benefit over (hidden width x input noise)."""
    noise = hidden_sweep["noise_levels"]
    widths = hidden_sweep["n_hidden_grid"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(hidden_sweep["benefit_peripheral"], cmap="RdBu_r",
                   vmin=-0.5, vmax=0.5, aspect="auto", origin="lower",
                   extent=[noise[0], noise[-1], 0, len(widths)])
    ax.set_yticks(np.arange(len(widths)) + 0.5)
    ax.set_yticklabels(widths)
    ax.set(xlabel="input noise std", ylabel="hidden units",
           title=f"Curriculum benefit (curr - std), {hidden_sweep['num_steps']} steps")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    return fig


def plot_hidden_benefit_lines(hidden_sweep):
    """Curriculum benefit vs input noise, one line per hidden width."""
    noise = hidden_sweep["noise_levels"]
    widths = hidden_sweep["n_hidden_grid"]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for i, h in enumerate(widths):
        ax.plot(noise, hidden_sweep["benefit_peripheral"][i], "-o",
                label=f"{h} units")
    ax.axhline(0.0, ls=":", c="gray")
    ax.set(xlabel="input noise std", ylabel="curriculum benefit (curr - std)",
           title="Curriculum benefit vs noise, by hidden width")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def plot_init_scale_x_noise(init_sweep, group="peripheral"):
    """Heatmap of final curriculum benefit over (hidden-init scale x noise).

    Small init scale = rich / feature-learning regime; large = lazy. ``group`` =
    'peripheral' (exceptions) or 'central' (rule)."""
    noise = init_sweep["noise_levels"]
    scales = init_sweep["init_scales"]
    benefit = init_sweep[f"benefit_{group}"]                 # (scale, noise)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(benefit, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto",
                   origin="lower", extent=[noise[0], noise[-1], 0, len(scales)])
    ax.set_yticks(np.arange(len(scales)) + 0.5)
    ax.set_yticklabels(scales)
    ax.set(xlabel="input noise std", ylabel="hidden-weight init scale",
           title=f"Curriculum benefit on {_GROUP_LABEL[group]} accuracy "
                 f"(curr - std), {init_sweep['num_steps']} steps, "
                 f"width={init_sweep['n_hidden']}")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    return fig


def plot_init_scale_accuracy(init_sweep, condition="standard", group="peripheral"):
    """Heatmap of raw final accuracy over (init scale x noise) for one condition.

    Useful to see the rich/lazy transition directly (not just the benefit)."""
    noise = init_sweep["noise_levels"]
    scales = init_sweep["init_scales"]
    acc = init_sweep[f"{condition}_{group}"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(acc, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto",
                   origin="lower", extent=[noise[0], noise[-1], 0, len(scales)])
    ax.set_yticks(np.arange(len(scales)) + 0.5)
    ax.set_yticklabels(scales)
    ax.set(xlabel="input noise std", ylabel="hidden-weight init scale",
           title=f"{condition}: final {_GROUP_LABEL[group]} accuracy")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    return fig
