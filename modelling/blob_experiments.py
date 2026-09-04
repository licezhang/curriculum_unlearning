"""Sweeps for the blob-curriculum question, built on ``train_blob_grid``.

Headline question: do small-random-init ReLU nets show a *stronger* curriculum
benefit when the inputs are Gaussian blobs (``signal_noise_std > 0``) than when
they are the 8 fixed examples (``std = 0``)? And how does that interact with the
length of training and the number of hidden units?

Two experiments cover it:
  * ``main_sweep``  -- one long run over noise levels at a fixed width, logging
    accuracy at many checkpoints. Gives benefit-vs-noise, the (noise x steps)
    interaction, and learning curves in a single pass.
  * ``hidden_sweep`` -- final benefit over (hidden width x noise level).
"""

from dataclasses import asdict, dataclass

import numpy as np

from blob_task import BlobTask
from blob_training import train_blob_grid


@dataclass
class BlobConfig:
    """All knobs for the blob-curriculum sweeps."""

    num_seeds: int = 256
    n_hidden: int = 16              # width used for the main sweep
    max_steps: int = 4000
    checkpoints: tuple = tuple(range(200, 4001, 200))   # eval schedule
    noise_levels: tuple = tuple(np.round(np.arange(0.0, 2.01, 0.25), 3))
    exception_prob: float = 0.25    # mean exception exposure (matched)
    learning_rate: float = 0.05
    weight_init_std: float = 0.5    # hidden layer: BOTH weights W and bias b
    small_init_std: float = 0.01    # readout default (v, and output bias c)
    readout_init_std: float = None  # None = follow small_init_std
    optimizer: str = "sgd"
    use_output_bias: bool = False   # readout/output bias removed by default
    accuracy_mode: str = "argmax"   # 'argmax' (hard) or 'expected' (soft/sigmoid)
    output_temperature: float = 1.0  # output unit is sigmoid(logit / T); 1 = as before

    n_hidden_grid: tuple = (2, 4, 8, 16, 32, 64)
    # hidden-weight init scale grid (small = rich/feature-learning, large = lazy)
    init_scale_grid: tuple = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    representative_noise: tuple = (0.0, 0.5, 1.0, 1.5, 2.0)  # learning-curve panels
    base_seed: int = 0

    def to_json_dict(self):
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _grid_kwargs(config: BlobConfig, task: BlobTask) -> dict:
    return dict(
        task=task,
        num_seeds=config.num_seeds,
        exception_prob=config.exception_prob,
        learning_rate=config.learning_rate,
        weight_init_std=config.weight_init_std,
        small_init_std=config.small_init_std,
        readout_init_std=config.readout_init_std,
        optimizer_name=config.optimizer,
        use_output_bias=config.use_output_bias,
        accuracy_mode=config.accuracy_mode,
        output_temperature=config.output_temperature,
        seed=config.base_seed,
    )


def main_sweep(config: BlobConfig, task: BlobTask) -> dict:
    """One run over noise levels with many checkpoints (fixed width)."""
    grid = train_blob_grid(
        noise_levels=config.noise_levels, n_hidden=config.n_hidden,
        num_steps=config.max_steps, checkpoints=config.checkpoints,
        **_grid_kwargs(config, task),
    )
    std_p, curr_p = grid["peripheral_accuracy"][:2]  # standard, curriculum (K, T)
    std_c, curr_c = grid["central_accuracy"][:2]
    return dict(
        noise_levels=grid["noise_levels"],
        checkpoints=grid["checkpoints"],
        n_hidden=config.n_hidden,
        num_seeds=config.num_seeds,
        weight_init_std=config.weight_init_std,
        exception_prob=config.exception_prob,
        representative_noise=np.array(config.representative_noise),
        standard_peripheral=std_p, curriculum_peripheral=curr_p,
        standard_central=std_c, curriculum_central=curr_c,
        benefit_peripheral=curr_p - std_p,            # (K, T)
        # BCE loss on the clean prototypes, per group (2, K, T) -> split by cond
        standard_peripheral_loss=grid["peripheral_loss"][0],
        curriculum_peripheral_loss=grid["peripheral_loss"][1],
        standard_central_loss=grid["central_loss"][0],
        curriculum_central_loss=grid["central_loss"][1],
        standard_overall_loss=grid["overall_loss"][0],
        curriculum_overall_loss=grid["overall_loss"][1],
    )


def compute_curve_data(config: BlobConfig, task: BlobTask, noise_levels=None,
                       n_hidden=None, weight_init_std=None, num_steps=None,
                       checkpoints=None, num_seeds=None, accuracy_mode=None,
                       record_rep_steps=None, rank=None, pr_target=None,
                       output_temperature=None, readout_init_std=None) -> dict:
    """Train one grid live and shape it like ``main_sweep`` for the curve plots.

    Lets the notebook choose the width, init scale, noise levels, steps and seeds
    that the learning-curve / loss-curve panels are drawn from (everything else
    -- optimizer, lr, output bias, exception prob -- comes from ``config``). The
    chosen ``noise_levels`` become the panels (``representative_noise``).

    ``rank`` (optional, 1..min(n_hidden,3)) makes the hidden weights exactly that
    rank at init (``PR(W0) = rank``) at matched Frobenius norm; ``None`` (default)
    is the usual full-rank init, so existing runs are unaffected."""
    noise_levels = config.noise_levels if noise_levels is None else noise_levels
    n_hidden = config.n_hidden if n_hidden is None else n_hidden
    weight_init_std = config.weight_init_std if weight_init_std is None else weight_init_std
    num_steps = config.max_steps if num_steps is None else num_steps
    num_seeds = config.num_seeds if num_seeds is None else num_seeds
    if checkpoints is None:
        # dense early (every 20 steps up to 400) so the rise off step 0 is
        # visible, then coarser across the rest of training
        early = range(0, min(400, num_steps) + 1, 20)
        coarse = range(0, num_steps + 1, max(1, num_steps // 30))
        checkpoints = sorted(set(early) | set(coarse) | {0, num_steps})

    kw = _grid_kwargs(config, task)
    kw["weight_init_std"] = weight_init_std
    kw["num_seeds"] = num_seeds
    if accuracy_mode is not None:
        kw["accuracy_mode"] = accuracy_mode
    if record_rep_steps is not None:
        kw["record_rep_steps"] = record_rep_steps
    if rank is not None:
        kw["rank"] = rank
    if pr_target is not None:
        kw["pr_target"] = pr_target
    if output_temperature is not None:
        kw["output_temperature"] = output_temperature
    if readout_init_std is not None:
        kw["readout_init_std"] = readout_init_std
    grid = train_blob_grid(noise_levels=noise_levels, n_hidden=n_hidden,
                           num_steps=num_steps, checkpoints=checkpoints, **kw)
    conds = grid["conditions"]
    out = dict(
        noise_levels=grid["noise_levels"], checkpoints=grid["checkpoints"],
        n_hidden=n_hidden, num_seeds=num_seeds, weight_init_std=weight_init_std,
        exception_prob=config.exception_prob,
        accuracy_mode=kw["accuracy_mode"], conditions=list(conds),
        representative_noise=np.array(list(noise_levels)), rank=rank,
        pr_target=pr_target,
        output_temperature=kw["output_temperature"],
        readout_init_std=kw["readout_init_std"],
    )
    # expose per-condition mean + SEM (over seeds) for each plotted group
    for dst, grid_key in (("peripheral", "peripheral_accuracy"),
                          ("central", "central_accuracy"),
                          ("peripheral_loss", "peripheral_loss"),
                          ("central_loss", "central_loss"),
                          ("overall_loss", "overall_loss"),
                          ("participation_ratio", "participation_ratio"),
                          ("active_units", "active_unit_fraction"),
                          ("linearity", "logit_linearity")):
        for ci, cond in enumerate(conds):
            out[f"{cond}_{dst}"] = grid[grid_key][ci]
            out[f"{cond}_{dst}_sem"] = grid[grid_key + "_sem"][ci]
            per_seed_key = grid_key + "_per_seed"
            if per_seed_key in grid:
                out[f"{cond}_{dst}_per_seed"] = grid[per_seed_key][ci]  # (K,S,T)
    # per-prototype logits over training: (K, 8, T) per condition (rule 0..5, exc 6..7)
    if "prototype_logits" in grid:
        for ci, cond in enumerate(conds):
            out[f"{cond}_prototype_logits"] = grid["prototype_logits"][ci]
            out[f"{cond}_prototype_logits_sem"] = grid["prototype_logits_sem"][ci]
    out["benefit_peripheral"] = out["curriculum_peripheral"] - out["standard_peripheral"]
    if "anti_curriculum_peripheral" in out:
        out["anti_benefit_peripheral"] = (
            out["anti_curriculum_peripheral"] - out["standard_peripheral"])
    if "rep_snapshots" in grid:
        out["rep_snapshots"] = grid["rep_snapshots"]      # {step: (C,P,8,H)}
        out["snapshots"] = sorted(grid["rep_snapshots"])
    return out


def hidden_sweep(config: BlobConfig, task: BlobTask) -> dict:
    """Final peripheral accuracy / curriculum benefit over (width x noise)."""
    widths = list(config.n_hidden_grid)
    K = len(config.noise_levels)
    std = np.zeros((len(widths), K))
    curr = np.zeros((len(widths), K))
    for i, h in enumerate(widths):
        grid = train_blob_grid(
            noise_levels=config.noise_levels, n_hidden=h,
            num_steps=config.max_steps, checkpoints=[config.max_steps],
            **_grid_kwargs(config, task),
        )
        std[i] = grid["peripheral_accuracy"][0, :, -1]     # (K,)
        curr[i] = grid["peripheral_accuracy"][1, :, -1]
    return dict(
        n_hidden_grid=np.array(widths),
        noise_levels=np.array(config.noise_levels),
        num_steps=config.max_steps,
        standard_peripheral=std, curriculum_peripheral=curr,
        benefit_peripheral=curr - std,                     # (H, K)
    )


def init_scale_sweep(config: BlobConfig, task: BlobTask) -> dict:
    """Final accuracy / curriculum benefit over (hidden-init scale x noise).

    Sweeps ``weight_init_std`` from small (rich / feature-learning regime) to
    large (lazy / ~NTK regime) at fixed width and step budget. Records both
    exception (peripheral) and rule (central) accuracy so we can see the
    rule-for-exception trade-off as a function of the init scale."""
    scales = list(config.init_scale_grid)
    K = len(config.noise_levels)
    shape = (len(scales), K)
    out_arrays = {k: np.zeros(shape) for k in
                  ("standard_peripheral", "curriculum_peripheral",
                   "standard_central", "curriculum_central")}
    for i, scale in enumerate(scales):
        kw = _grid_kwargs(config, task)
        kw["weight_init_std"] = scale
        grid = train_blob_grid(
            noise_levels=config.noise_levels, n_hidden=config.n_hidden,
            num_steps=config.max_steps, checkpoints=[config.max_steps], **kw,
        )
        out_arrays["standard_peripheral"][i] = grid["peripheral_accuracy"][0, :, -1]
        out_arrays["curriculum_peripheral"][i] = grid["peripheral_accuracy"][1, :, -1]
        out_arrays["standard_central"][i] = grid["central_accuracy"][0, :, -1]
        out_arrays["curriculum_central"][i] = grid["central_accuracy"][1, :, -1]
    return dict(
        init_scales=np.array(scales),
        noise_levels=np.array(config.noise_levels),
        num_steps=config.max_steps, n_hidden=config.n_hidden,
        benefit_peripheral=out_arrays["curriculum_peripheral"] - out_arrays["standard_peripheral"],
        benefit_central=out_arrays["curriculum_central"] - out_arrays["standard_central"],
        **out_arrays,
    )
