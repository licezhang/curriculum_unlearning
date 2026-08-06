"""Batched training of small-random-init ReLU MLPs on the blob task.

``train_blob_grid`` trains, in one vectorised run, a whole population of nets
indexed by (condition x noise-level x seed) for a fixed hidden width and step
budget. Each net is an ``n_hidden``-unit ReLU MLP (readout + output bias) started
from small random weights, streaming one Gaussian-blob sample per step. Standard
vs curriculum differ only in the exception-exposure *probability schedule* (both
have the same mean exposure); the per-step input noise, prototype draws and init
are shared across the two conditions so the comparison is paired.

Accuracy is evaluated on the clean 8 prototypes at each requested checkpoint, so
one long run yields the whole "vs length-of-training" curve for free.
"""

import numpy as np
import torch
import torch.nn.functional as F

from blob_task import BlobTask


def build_optimizer(name, params, learning_rate):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    raise ValueError(f"unknown optimizer {name!r} (use 'sgd' or 'adam')")


DEFAULT_CONDITIONS = ("standard", "curriculum", "anti_curriculum")


def _condition_schedule(name, exception_prob, num_steps, device):
    """Per-step exception probability for one named condition (matched mean p).

    * ``standard``        -- constant ``exception_prob``.
    * ``curriculum``      -- simple->complex: ramps 0 -> 2*exception_prob.
    * ``anti_curriculum`` -- complex->simple: ramps 2*exception_prob -> 0 (the
      exact mirror of ``curriculum``).

    All three have mean ``exception_prob`` (matched total exposure), differing
    only in *when* exceptions appear."""
    p = float(exception_prob)
    frac = torch.linspace(0.0, 1.0, num_steps, device=device)   # 0..1 over training
    if name == "standard":
        return torch.full((num_steps,), p, device=device)
    if name == "curriculum":
        return (2.0 * p * frac).clamp(0.0, 1.0)
    if name == "anti_curriculum":
        return (2.0 * p * (1.0 - frac)).clamp(0.0, 1.0)
    if name == "curriculum_hold":
        # Control for `curriculum`'s *terminal* exposure rate: ramp 0 -> 2p over
        # the first half (mean p there), then hold at p. Mean is still p (matched
        # like the others), but training now ENDS at p, exactly as `standard`
        # does -- whereas `curriculum` ends at 2p, i.e. half of the last trials
        # are exceptions. Separates "rule-first ordering" from "finishes on an
        # exception-heavy diet", which matters because online SGD is recency-
        # dominated (no replay).
        ramp = 4.0 * p * frac
        return torch.where(frac < 0.5, ramp,
                           torch.full_like(frac, p)).clamp(0.0, 1.0)
    raise ValueError(f"unknown condition {name!r} (standard / curriculum / "
                     f"anti_curriculum / curriculum_hold)")


def low_rank_init_W(population, n_hidden, input_dim, rank, weight_init_std,
                    generator, device):
    """Exactly-rank-``r`` hidden weights for a population of nets: (P, H, d).

    ``W = A @ B.T`` with ``A (H,r)`` / ``B (d,r)`` having random orthonormal
    columns (QR of Gaussians), so all ``r`` singular values are equal and
    ``PR(W) = r`` exactly. The result is rescaled to the *same* Frobenius norm a
    full-rank draw would have under this module's convention
    (``W ~ randn * weight_init_std`` => ``||W||_F = weight_init_std*sqrt(H*d)``),
    so varying ``rank`` changes only the shape of the spectrum, not the overall
    init scale."""
    A = torch.randn(population, n_hidden, rank, generator=generator, device=device)
    B = torch.randn(population, input_dim, rank, generator=generator, device=device)
    Aq, _ = torch.linalg.qr(A)                        # (P,H,r), orthonormal cols
    Bq, _ = torch.linalg.qr(B)                        # (P,d,r), orthonormal cols
    W = Aq @ Bq.transpose(-1, -2)                     # (P,H,d), ||W||_F = sqrt(r)
    target = weight_init_std * (n_hidden * input_dim) ** 0.5
    cur = W.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return W * (target / cur)


def spectrum_for_pr(target_pr, dim, max_alpha=80.0, iters=200):
    """Singular-value profile ``(dim,)``, unit norm, with ``PR(s) == target_pr``.

    Uses a power-law profile ``s_i ~ i**-alpha`` and bisects on ``alpha``: the
    participation ratio ``(sum s^2)^2 / sum s^4`` decreases monotonically from
    ``dim`` (alpha = 0, all values equal) toward 1 (alpha -> inf, one direction),
    so any target in ``[1, dim]`` is reachable. This is what turns the integer
    ``rank`` knob into a *continuous* compression axis."""
    dim = int(dim)
    target_pr = float(target_pr)
    if not (1.0 - 1e-9 <= target_pr <= dim + 1e-9):
        raise ValueError(f"target_pr must lie in [1, {dim}], got {target_pr}")
    idx = np.arange(1, dim + 1, dtype=np.float64)
    if target_pr >= dim - 1e-12:
        s = np.ones(dim)
        return s / np.linalg.norm(s)

    def pr_of(alpha):
        s2 = (idx ** (-alpha)) ** 2
        return (s2.sum() ** 2) / (s2 ** 2).sum()

    lo, hi = 0.0, max_alpha                    # pr_of is decreasing in alpha
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if pr_of(mid) > target_pr:
            lo = mid
        else:
            hi = mid
    s = idx ** (-0.5 * (lo + hi))
    return s / np.linalg.norm(s)


def shaped_init_W(population, n_hidden, input_dim, target_pr, weight_init_std,
                  generator, device):
    """Hidden weights (P, H, d) with a *continuous* participation ratio.

    ``W = A diag(s) B.T`` with random orthonormal ``A, B`` (so the singular
    values are exactly ``s``) and ``s`` shaped by :func:`spectrum_for_pr`, then
    rescaled to the full-rank Frobenius norm ``weight_init_std*sqrt(H*d)``. Unlike
    the integer ``rank`` init this keeps full support with a decaying spectrum, so
    ``PR(W0)`` can be set to any value in ``[1, min(H, d)]``."""
    r = min(n_hidden, input_dim)
    s = torch.tensor(spectrum_for_pr(target_pr, r), dtype=torch.float32,
                     device=device)                             # (r,), unit norm
    A = torch.randn(population, n_hidden, r, generator=generator, device=device)
    B = torch.randn(population, input_dim, r, generator=generator, device=device)
    Aq, _ = torch.linalg.qr(A)                        # (P,H,r)
    Bq, _ = torch.linalg.qr(B)                        # (P,d,r)
    W = (Aq * s) @ Bq.transpose(-1, -2)               # (P,H,d), sing. values = s
    target = weight_init_std * (n_hidden * input_dim) ** 0.5
    cur = W.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return W * (target / cur)


def random_init_state(population, n_hidden, weight_init_std, small_init_std,
                      use_output_bias, generator, device, rank=None,
                      readout_init_std=None, pr_target=None):
    """Small random init for a population of two-layer ReLU nets.

    Returns ``{'W0':(P,H,3), 'b0':(P,H), 'v0':(P,H), 'c0':(P,)}`` -- the un-
    expanded (per-net) initial state that ``train_blob_grid`` broadcasts across
    conditions. Draw order (W0, b0, v0, then c0 only if ``use_output_bias``)
    matches the previous inline init, so behaviour is unchanged when a fresh
    state is used. Also the exchange format for pretraining -> transfer.

    Two layers, two scales. The hidden layer -- weights ``W0`` *and* bias ``b0``,
    which belong to the same layer -- is initialised at ``weight_init_std``. The
    readout ``v0`` (and output bias ``c0``) is initialised at
    ``readout_init_std``, falling back to ``small_init_std`` when that is None.

    Two ways to bias the init spectrum (both at matched Frobenius norm, so only
    the *shape* of the spectrum changes):

    * ``pr_target`` -- continuous participation ratio in ``[1, min(n_hidden, 3)]``
      via a shaped, full-support spectrum (:func:`shaped_init_W`). Takes
      precedence over ``rank``.
    * ``rank`` -- integer hard rank ``< min(n_hidden, 3)``; ``PR(W0) = rank``
      exactly with equal singular values (:func:`low_rank_init_W`).

    ``None`` for both (default) keeps the ordinary full-rank Gaussian draw,
    byte-for-byte as before."""
    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device)
    if readout_init_std is None:
        readout_init_std = small_init_std
    max_rank = min(n_hidden, 3)
    if pr_target is not None:
        W0 = shaped_init_W(population, n_hidden, 3, pr_target, weight_init_std,
                           generator, device)
    elif rank is None or int(rank) >= max_rank:
        W0 = randn(population, n_hidden, 3) * weight_init_std
    else:
        if int(rank) < 1:
            raise ValueError(f"rank must be >= 1, got {rank!r}")
        W0 = low_rank_init_W(population, n_hidden, 3, int(rank), weight_init_std,
                             generator, device)
    b0 = randn(population, n_hidden) * weight_init_std   # same layer as W0
    v0 = randn(population, n_hidden) * readout_init_std
    c0 = (randn(population) * readout_init_std if use_output_bias
          else torch.zeros(population, device=device))
    return {"W0": W0, "b0": b0, "v0": v0, "c0": c0}


def train_blob_grid(
    task: BlobTask,
    noise_levels,
    n_hidden: int = 16,
    num_steps: int = 4000,
    checkpoints=None,
    num_seeds: int = 256,
    exception_prob: float = 0.25,
    learning_rate: float = 0.05,
    weight_init_std: float = 0.5,
    small_init_std: float = 0.01,
    optimizer_name: str = "sgd",
    use_output_bias: bool = True,
    accuracy_mode: str = "argmax",
    conditions=DEFAULT_CONDITIONS,
    record_rep_steps=None,
    init_state=None,
    return_final_per_seed=False,
    rank=None,
    pr_target=None,
    readout_init_std=None,
    output_temperature: float = 1.0,
    seed: int = 0,
) -> dict:
    """Train the (condition x noise x seed) population and log checkpoint accuracy.

    Parameters
    ----------
    noise_levels : iterable of ``signal_noise_std`` values (0 = fixed 8 examples).
    n_hidden : number of hidden ReLU units.
    num_steps : total online SGD steps (batch size 1 per net).
    checkpoints : update counts at which to evaluate (default: just ``num_steps``).
    num_seeds : nets averaged per (condition, noise) cell.
    exception_prob : mean exception probability (both conditions matched).
    weight_init_std : std of the random init for the whole hidden layer -- both
        the weights ``W`` and the bias ``b`` (they belong to the same layer).
    small_init_std : std of the readout init (``v``, and the output bias ``c``),
        used when ``readout_init_std`` is not given.
    use_output_bias : if False, the readout (output) bias ``c`` is fixed at 0 and
        not learned -- the logit is purely ``sum_h v_h * relu(w_h . x + b_h)``.
    accuracy_mode : 'argmax' (hard, threshold the logit at 0) or 'expected' (soft
        -- the probability the sigmoid assigns to the *true* label, so confidence
        counts). Only the accuracy metrics change; the BCE loss is unaffected.
    conditions : ordered names of the exposure schedules to run in parallel
        (default standard / curriculum / anti_curriculum). All share the same
        paired init and stimulus stream, differing only in the schedule.
    init_state : optional ``{'W0','b0','v0','c0'}`` (per-net, shape (P,H,3)/(P,H)/
        (P,H)/(P,)) to start from instead of a fresh random init -- used to carry
        pretrained weights into a transfer phase. Broadcast identically across
        conditions (preserving the paired-init property).
    return_final_per_seed : if True, also return ``final_peripheral_per_seed`` and
        ``final_central_per_seed`` of shape (C,K,S) from the last checkpoint (for
        per-seed benefit / correlation analyses).
    rank : optional exact rank of the hidden weights at init (1..min(n_hidden,3)).
        ``PR(W0) = rank`` exactly, at the same Frobenius norm as a full-rank draw,
        so only the spectrum shape changes. ``None`` (default) = full-rank
        Gaussian init, identical to before. Ignored when ``init_state`` is given.
    pr_target : optional *continuous* participation ratio of the hidden weights
        at init, in ``[1, min(n_hidden, 3)]`` (shaped full-support spectrum at
        matched Frobenius norm). Takes precedence over ``rank``; ``None``
        (default) leaves the init unchanged. Ignored when ``init_state`` is given.
    readout_init_std : optional separate init scale for the readout ``v`` (and
        output bias ``c``). ``None`` (default) makes it follow ``small_init_std``
        (previous behaviour, bit-identical). Lets the readout be initialised
        independently of the hidden bias. Ignored when ``init_state`` is given.
    output_temperature : temperature ``T`` of the output sigmoid, i.e. the unit is
        ``sigmoid(logit / T)``. Applies to the *training* loss as well as the
        metrics, so it changes learning dynamics, not just the readout: ``T > 1``
        softens (smaller effective logits, later saturation, weaker gradients),
        ``T < 1`` sharpens. Default ``1.0`` reproduces previous behaviour exactly.
        Caveat: ``argmax`` accuracy is invariant to ``T`` only when *evaluating a
        fixed network* (``T`` does not change the sign of the logit) -- across
        runs it still moves a lot, because ``T`` changes what the net learns. The
        readout can partly absorb ``T`` by growing ``v``, so the effect is
        strongest at a fixed step/learning-rate budget.

    Returns
    -------
    dict with ``noise_levels`` (K,), ``checkpoints`` (T,), ``conditions`` (C
    names), ``n_hidden``, and per-metric arrays of shape (C, K, T):
    ``peripheral_accuracy`` (exceptions), ``central_accuracy`` (rule),
    ``overall_accuracy`` (+ losses, participation_ratio, active_unit_fraction),
    each with a ``_sem`` companion.
    """
    if accuracy_mode not in ("argmax", "expected"):
        raise ValueError(f"accuracy_mode must be 'argmax' or 'expected', got {accuracy_mode!r}")
    if not output_temperature > 0:
        raise ValueError(f"output_temperature must be > 0, got {output_temperature!r}")
    device = task.device
    conditions = list(conditions)
    C = len(conditions)
    noise_levels = np.asarray(noise_levels, dtype=np.float32)
    if checkpoints is None:
        checkpoints = [num_steps]
    # always include step 0 (the initial state) so curves start from the very start
    checkpoints = sorted({0} | {int(c) for c in checkpoints if 0 <= int(c) <= num_steps})
    checkpoint_set = set(checkpoints)
    K, S = len(noise_levels), num_seeds
    population = K * S                                   # paired across conditions
    g = torch.Generator(device=device).manual_seed(seed)

    noise_per_net = torch.tensor(
        np.repeat(noise_levels, S), device=device)       # (P,)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device)

    # ----- init, shared (paired) across the conditions -----
    if init_state is None:                                # fresh small random init
        init_state = random_init_state(population, n_hidden, weight_init_std,
                                       small_init_std, use_output_bias, g, device,
                                       rank=rank, pr_target=pr_target,
                                       readout_init_std=readout_init_std)
    if init_state["W0"].shape[0] != population or init_state["W0"].shape[1] != n_hidden:
        raise ValueError(f"init_state W0 shape {tuple(init_state['W0'].shape)} does "
                         f"not match (population={population}, n_hidden={n_hidden})")
    W = init_state["W0"].to(device).unsqueeze(0).expand(C, -1, -1, -1).clone().requires_grad_(True)
    b = init_state["b0"].to(device).unsqueeze(0).expand(C, -1, -1).clone().requires_grad_(True)
    v = init_state["v0"].to(device).unsqueeze(0).expand(C, -1, -1).clone().requires_grad_(True)
    params = [W, b, v]
    if use_output_bias:
        c = init_state["c0"].to(device).unsqueeze(0).expand(C, -1).clone().requires_grad_(True)
        params.append(c)
    else:
        c = torch.zeros(C, population, device=device)     # fixed 0, not learned
    optimizer = build_optimizer(optimizer_name, params, learning_rate)

    p_sched = torch.stack([_condition_schedule(n, exception_prob, num_steps, device)
                           for n in conditions], 0)         # (C, T)
    n_central = task.central_prototypes.shape[0]
    n_periph = task.peripheral_prototypes.shape[0]

    protos = task.all_prototypes                          # (8, 3)
    labels = task.all_labels                              # (8,)
    is_periph = task.is_peripheral
    # affine design matrix [1, x0, x1, x2] for the logit-linearity R^2 metric
    design = torch.cat([torch.ones(8, 1, device=device), protos], 1)   # (8, 4)
    design_pinv = torch.linalg.pinv(design)                            # (4, 8)

    metric_keys = ("peripheral_accuracy", "central_accuracy", "overall_accuracy",
                   "peripheral_loss", "central_loss", "overall_loss",
                   "participation_ratio", "active_unit_fraction",
                   "logit_linearity")
    metrics = {k: [] for k in metric_keys}
    metrics_sem = {k: [] for k in metric_keys}   # std-error over seeds per cell
    final_per_seed = {}                          # last-checkpoint (C,K,S) accuracy
    logit_trace, logit_trace_sem = [], []        # per-prototype logits -> (C,K,8,T)
    targets = labels.view(1, 1, -1)                        # broadcast to (2,P,8)

    def evaluate():
        # accuracy AND BCE loss on the clean 8 prototypes (the blob centres)
        with torch.no_grad():
            pre = torch.einsum("nd,cphd->cphn", protos, W) + b.unsqueeze(-1)
            hidden = torch.relu(pre)                       # (2,P,H,8)
            logits = torch.einsum("cphn,cph->cpn", hidden, v) + c.unsqueeze(-1)
            logits = logits / output_temperature           # sigmoid(logit / T)
            if accuracy_mode == "expected":
                probs = torch.sigmoid(logits)              # (2,P,8)
                correct = labels * probs + (1 - labels) * (1 - probs)  # P(true label)
            else:
                correct = ((logits > 0).float() == labels).float()
            bce = F.binary_cross_entropy_with_logits(
                logits, targets.expand_as(logits), reduction="none")  # (2,P,8)
            per_net = {
                "peripheral_accuracy": correct[:, :, is_periph].mean(-1),
                "central_accuracy": correct[:, :, ~is_periph].mean(-1),
                "overall_accuracy": correct.mean(-1),
                "peripheral_loss": bce[:, :, is_periph].mean(-1),
                "central_loss": bce[:, :, ~is_periph].mean(-1),
                "overall_loss": bce.mean(-1),
            }
            # --- mechanism metrics on the 8 centroids' post-ReLU representation ---
            # participation ratio (centered, all 8): effective dimensionality ~1..7
            reps = hidden.permute(0, 1, 3, 2)              # (2,P,8,H) activations
            reps = reps - reps.mean(dim=2, keepdim=True)   # centre over the 8 points
            lam = torch.linalg.svdvals(reps) ** 2          # (2,P,8) covariance eigvals
            per_net["participation_ratio"] = (
                lam.sum(-1) ** 2 / lam.pow(2).sum(-1).clamp_min(1e-12))     # (2,P)
            # fraction of hidden units that fire on >= 1 centroid (recruitment)
            per_net["active_unit_fraction"] = (
                (hidden > 0).any(dim=3).float().mean(dim=2))               # (2,P)
            # decision-boundary linearity: R^2 of an affine fit to the logits on
            # the 8 centroids (1 = linear/rule-only; <1 = boundary bends for pocket)
            beta = torch.einsum("ab,cpb->cpa", design_pinv, logits)       # (C,P,4)
            pred_lin = torch.einsum("sb,cpb->cps", design, beta)          # (C,P,8)
            ss_res = ((logits - pred_lin) ** 2).sum(-1)
            ss_tot = ((logits - logits.mean(-1, keepdim=True)) ** 2).sum(-1).clamp_min(1e-8)
            per_net["logit_linearity"] = (1 - ss_res / ss_tot).clamp(-1, 1)  # (C,P)
            for key, arr in per_net.items():               # (C,P) -> (C,K)
                grouped = arr.reshape(C, K, S)
                metrics[key].append(grouped.mean(-1).cpu().numpy())
                sem = grouped.std(-1, unbiased=False) / (S ** 0.5)
                metrics_sem[key].append(sem.cpu().numpy())
            if return_final_per_seed:                       # overwritten -> holds last
                final_per_seed["peripheral"] = (
                    per_net["peripheral_accuracy"].reshape(C, K, S).cpu().numpy())
                final_per_seed["central"] = (
                    per_net["central_accuracy"].reshape(C, K, S).cpu().numpy())
            # per-prototype decision variable (post-temperature), averaged over seeds
            by_seed = logits.reshape(C, K, S, -1)           # (C,K,S,8)
            logit_trace.append(by_seed.mean(2).cpu().numpy())               # (C,K,8)
            logit_trace_sem.append(
                (by_seed.std(2, unbiased=False) / (S ** 0.5)).cpu().numpy())

    # optional: snapshot the 8 centroids' post-ReLU reps at chosen steps (PCA)
    rep_step_set = {int(s) for s in record_rep_steps} if record_rep_steps else set()
    rep_snapshots = {}

    def snapshot_reps(step):
        with torch.no_grad():
            h = torch.relu(torch.einsum("nd,cphd->cphn", protos, W)
                           + b.unsqueeze(-1))            # (C,P,H,8)
            rep_snapshots[int(step)] = h.permute(0, 1, 3, 2).cpu().numpy()  # (C,P,8,H)

    if 0 in checkpoint_set:
        evaluate()                                        # initial state (step 0)
    if 0 in rep_step_set:
        snapshot_reps(0)
    for t in range(num_steps):
        u = torch.rand(population, generator=g, device=device)
        rule_idx = torch.randint(0, n_central, (population,), generator=g, device=device)
        exc_idx = torch.randint(0, n_periph, (population,), generator=g, device=device)
        noise = randn(population, 3) * noise_per_net.unsqueeze(1)     # (P,3)

        is_exc = u.unsqueeze(0) < p_sched[:, t].unsqueeze(1)          # (2,P)
        rule_x = task.central_prototypes[rule_idx]                   # (P,3)
        exc_x = task.peripheral_prototypes[exc_idx]
        base = torch.where(is_exc.unsqueeze(-1),
                           exc_x.unsqueeze(0), rule_x.unsqueeze(0))   # (2,P,3)
        x = base + noise.unsqueeze(0)
        y = torch.where(is_exc, task.peripheral_labels[exc_idx].unsqueeze(0),
                        task.central_labels[rule_idx].unsqueeze(0))   # (2,P)

        optimizer.zero_grad()
        pre = torch.einsum("cpd,cphd->cph", x, W) + b                 # (2,P,H)
        logit = ((torch.relu(pre) * v).sum(-1) + c) / output_temperature   # (2,P)
        F.binary_cross_entropy_with_logits(logit, y, reduction="sum").backward()
        optimizer.step()

        if (t + 1) in checkpoint_set:
            evaluate()
        if (t + 1) in rep_step_set:
            snapshot_reps(t + 1)

    out = dict(noise_levels=noise_levels, checkpoints=np.array(checkpoints),
               conditions=list(conditions), n_hidden=n_hidden)
    for key in metrics:
        out[key] = np.stack(metrics[key], axis=-1)        # (C, K, T) mean
        out[key + "_sem"] = np.stack(metrics_sem[key], axis=-1)  # (C, K, T) SEM
    if logit_trace:
        # decision variable per prototype over training (rule 0..5, exception 6..7)
        out["prototype_logits"] = np.stack(logit_trace, axis=-1)          # (C,K,8,T)
        out["prototype_logits_sem"] = np.stack(logit_trace_sem, axis=-1)
    if rep_snapshots:
        out["rep_snapshots"] = rep_snapshots              # {step: (C,P,8,H)}
    if return_final_per_seed and final_per_seed:
        out["final_peripheral_per_seed"] = final_per_seed["peripheral"]  # (C,K,S)
        out["final_central_per_seed"] = final_per_seed["central"]        # (C,K,S)
    return out
