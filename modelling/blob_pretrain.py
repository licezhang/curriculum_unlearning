"""Experiment 2: pretraining -> rule+exception (type-5) transfer.

Human Experiment 2: learners given a *compressed, low-dimensional prior* (a 1-D
rule) benefit more from a simple->complex curriculum on the type-5 task than
learners given a higher-dimensional prior (XOR). This module is the neural-net
analog, built on the same blob machinery (``train_blob_grid``) so Figures 2/3
are untouched:

  1. Start a population of small-random-init ReLU MLPs from one *shared* init.
  2. **Pretrain** each net on a **1-D** or **XOR** labelling of the *same* 3
     features (random feature(s) per seed), carrying over **all** weights (or,
     with ``reset_readout``, only the hidden features).
  3. **Transfer**: continue training on the type-5 task under **standard vs
     curriculum**, tracking exception (peripheral) accuracy and the effective
     dimensionality (participation ratio) of the representation.

Hypothesis to reproduce: 1-D pretraining -> lower PR (more compressed) -> larger
curriculum benefit on the exceptions than XOR or from-scratch.

Phase 1 needs a *dedicated* loop (not ``train_blob_grid``) because the labels are
per-seed (features chosen per net); phase 2 reuses the kernel via ``init_state``.
"""

import numpy as np
import torch
import torch.nn.functional as F

from blob_training import (_condition_schedule, build_optimizer,
                           random_init_state, train_blob_grid)

try:  # keep figures consistent with the rest of the notebook
    from blob_plotting import _COND_COLOR, _COND_LABEL
except Exception:                                                # pragma: no cover
    _COND_COLOR = {"standard": "C0", "curriculum": "C1"}
    _COND_LABEL = {"standard": "standard", "curriculum": "curriculum"}

_KIND_LABEL = {"none": "from scratch", "1d": "1-D rule", "xor": "XOR"}
_KIND_COLOR = {"none": "0.5", "1d": "#d1495b", "xor": "#3f88c5"}


def _try_format_axis():
    """The repo's ``format_axis`` if importable, else a no-op (module is usable
    outside the notebook)."""
    try:
        import sys
        if "/mnt/big/jirko/simplicitybias" not in sys.path:
            sys.path.insert(0, "/mnt/big/jirko/simplicitybias")
        from simplicitybias.plot_utils import format_axis
        return format_axis
    except Exception:                                            # pragma: no cover
        return lambda ax, *a, **k: None


# --------------------------------------------------------------------------- #
# Phase 1: pretraining tasks + dedicated training loop
# --------------------------------------------------------------------------- #
def pretrain_feature_labels(kind, num_seeds, prototypes, generator):
    """Per-seed labels for a 1-D / XOR pretraining task over the 8 cube corners.

    Features are assigned **randomly per seed** (the confirmed design), so every
    net sees a different low-dimensional prior.

    * ``1d``  : ``label = sign(x[feat])`` for a random ``feat in {0,1,2}``.
    * ``xor`` : ``label = sign(x[i]) != sign(x[j])`` for a random pair ``{i,j}``.

    Returns ``labels`` (S, 8) in {0,1}, ``features`` (S,) or (S, 2) of the chosen
    feature index/indices, and ``overlaps_rule`` (S,) bool = whether the assigned
    feature(s) include feature 0 (the rule axis ``y = NOT x0``)."""
    device = prototypes.device
    sign = (prototypes > 0).to(torch.long)                 # (8, 3) in {0,1}
    S = int(num_seeds)
    if kind == "1d":
        feat = torch.randint(0, 3, (S,), generator=generator, device=device)  # (S,)
        labels = sign[:, feat].t().to(torch.float32)       # (S, 8)
        overlaps_rule = (feat == 0)
        return labels, feat, overlaps_rule
    if kind == "xor":
        i = torch.randint(0, 3, (S,), generator=generator, device=device)
        off = torch.randint(1, 3, (S,), generator=generator, device=device)
        j = (i + off) % 3                                  # j != i
        si = sign[:, i].t()                                # (S, 8)
        sj = sign[:, j].t()
        labels = (si != sj).to(torch.float32)              # XOR
        overlaps_rule = (i == 0) | (j == 0)
        return labels, torch.stack([i, j], 1), overlaps_rule
    raise ValueError(f"unknown pretrain kind {kind!r} (use '1d' or 'xor')")


def pretrain_population(task, kind, init_state, num_steps, noise, learning_rate,
                        optimizer_name, use_output_bias, generator,
                        checkpoints=None, accuracy_mode="argmax",
                        output_temperature=1.0):
    """Pretrain a population of ``P`` nets (one per seed) on a 1-D / XOR task.

    Dedicated phase-1 loop: at each step every net draws a random corner of the
    cube plus Gaussian jitter of std ``noise`` and does BCE-SGD on its own
    per-seed label table. Same two-layer ReLU forward as the kernel. At each
    checkpoint, clean-corner accuracy, BCE loss and participation ratio are
    logged (mean +/-1 SEM over seeds) so the pretraining phase itself can be
    plotted over training.

    Returns ``{'state': {W0,b0,v0,c0}, 'pretrain_pr': (P,), 'pretrain_accuracy':
    (P,), 'overlaps_rule': (P,), 'features': ..., 'curve': {...}}`` -- ``state``
    is the per-net pretrained init to feed into ``train_blob_grid`` for the
    transfer phase; ``curve`` holds the over-training accuracy/loss/PR."""
    device = task.device
    protos = task.all_prototypes                           # (8, 3)
    n_proto = protos.shape[0]
    P = init_state["W0"].shape[0]
    if checkpoints is None:
        checkpoints = _default_checkpoints(num_steps)
    checkpoint_set = {int(s) for s in checkpoints if 0 <= int(s) <= num_steps}

    labels, features, overlaps_rule = pretrain_feature_labels(
        kind, P, protos, generator)                        # labels (P, 8)

    W = init_state["W0"].to(device).clone().requires_grad_(True)   # (P,H,a)
    b = init_state["b0"].to(device).clone().requires_grad_(True)   # (P,H)
    v = init_state["v0"].to(device).clone().requires_grad_(True)   # (P,H)
    params = [W, b, v]
    # optional linear input adapter A (P,a,3): pretraining learns through it, but
    # it is *not* carried over -- the transfer phase gets a fresh, reset adapter
    A = None
    if "A0" in init_state:
        A = init_state["A0"].to(device).clone().requires_grad_(True)   # (P,a,3)
        params.append(A)
    if use_output_bias:
        c = init_state["c0"].to(device).clone().requires_grad_(True)   # (P,)
        params.append(c)
    else:
        c = torch.zeros(P, device=device)
    optimizer = build_optimizer(optimizer_name, params, learning_rate)

    def evaluate():
        """Per-net (accuracy, BCE loss, PR) on the clean 8 corners -> (P,) each."""
        with torch.no_grad():
            if A is not None:
                lat = torch.einsum("nd,pad->pna", protos, A)           # (P,8,a)
            else:
                lat = protos.unsqueeze(0).expand(P, -1, -1)            # (P,8,3)
            pre = torch.einsum("pna,pha->phn", lat, W) + b.unsqueeze(-1)  # (P,H,8)
            hidden = torch.relu(pre)
            logits = (torch.einsum("phn,ph->pn", hidden, v)
                      + c.unsqueeze(-1)) / output_temperature          # (P,8)
            if accuracy_mode == "expected":                # soft: P(true label)
                probs = torch.sigmoid(logits)
                acc = (labels * probs + (1 - labels) * (1 - probs)).mean(-1)  # (P,)
            else:                                          # argmax: hard 0/1
                acc = ((logits > 0).float() == labels).float().mean(-1)      # (P,)
            bce = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none").mean(-1)                   # (P,)
            reps = hidden.permute(0, 2, 1)                 # (P,8,H)
            reps = reps - reps.mean(1, keepdim=True)       # centre over 8 corners
            lam = torch.linalg.svdvals(reps) ** 2          # (P, min(8,H))
            pr = lam.sum(-1) ** 2 / lam.pow(2).sum(-1).clamp_min(1e-12)      # (P,)
        return acc, bce, pr

    cpts, last = [], {}
    curve = {k: [] for k in ("accuracy", "loss", "participation_ratio")}
    curve_sem = {k: [] for k in curve}

    def record(step):
        acc, bce, pr = evaluate()
        last["acc"], last["pr"] = acc, pr
        for key, arr in (("accuracy", acc), ("loss", bce), ("participation_ratio", pr)):
            curve[key].append(arr.mean().item())
            curve_sem[key].append((arr.std(unbiased=False) / (P ** 0.5)).item())
        cpts.append(int(step))

    rows = torch.arange(P, device=device)
    if 0 in checkpoint_set:
        record(0)
    for t in range(num_steps):
        idx = torch.randint(0, n_proto, (P,), generator=generator, device=device)
        x = protos[idx] + noise * torch.randn(P, 3, generator=generator, device=device)
        y = labels[rows, idx]                              # (P,) per-net label
        optimizer.zero_grad()
        latent = torch.einsum("pd,pad->pa", x, A) if A is not None else x  # (P,a)
        pre = torch.einsum("pa,pha->ph", latent, W) + b    # (P,H)
        logit = ((torch.relu(pre) * v).sum(-1) + c) / output_temperature   # (P,)
        F.binary_cross_entropy_with_logits(logit, y, reduction="sum").backward()
        optimizer.step()
        if (t + 1) in checkpoint_set:
            record(t + 1)

    with torch.no_grad():
        state = {k: t.detach().clone()
                 for k, t in (("W0", W), ("b0", b), ("v0", v), ("c0", c))}
    curve_out = {"checkpoints": np.asarray(cpts)}
    for key in curve:
        curve_out[key] = np.asarray(curve[key])
        curve_out[key + "_sem"] = np.asarray(curve_sem[key])
    return dict(state=state, pretrain_pr=last["pr"].cpu().numpy(),
                pretrain_accuracy=last["acc"].cpu().numpy(),
                overlaps_rule=overlaps_rule.cpu().numpy(),
                features=features.cpu().numpy(), curve=curve_out)


# --------------------------------------------------------------------------- #
# Phase 2: compose pretraining -> transfer and shape like ``compute_curve_data``
# --------------------------------------------------------------------------- #
_CURVE_GROUPS = (("peripheral", "peripheral_accuracy"),
                 ("central", "central_accuracy"),
                 ("peripheral_loss", "peripheral_loss"),
                 ("central_loss", "central_loss"),
                 ("overall_loss", "overall_loss"),
                 ("participation_ratio", "participation_ratio"),
                 ("active_units", "active_unit_fraction"),
                 ("linearity", "logit_linearity"))


def _default_checkpoints(num_steps):
    early = range(0, min(400, num_steps) + 1, 20)
    coarse = range(0, num_steps + 1, max(1, num_steps // 30))
    return sorted(set(early) | set(coarse) | {0, num_steps})


def _shape_curve(grid):
    """One kind's ``train_blob_grid`` output (single noise level, conds standard/
    curriculum) reshaped to the ``{cond}_{group}[_sem]`` keys the plots expect,
    plus the final per-seed peripheral accuracy (C, S) for the benefit stats."""
    conds = grid["conditions"]
    out = {"conditions": list(conds), "checkpoints": grid["checkpoints"]}
    for dst, gkey in _CURVE_GROUPS:
        if gkey not in grid:                                   # adapter path may skip some
            continue
        for ci, cond in enumerate(conds):
            out[f"{cond}_{dst}"] = grid[gkey][ci, 0]           # (T,), noise idx 0
            out[f"{cond}_{dst}_sem"] = grid[gkey + "_sem"][ci, 0]
    out["benefit_peripheral"] = out["curriculum_peripheral"] - out["standard_peripheral"]
    fp = grid["final_peripheral_per_seed"][:, 0]               # (C, S)
    return out, fp


# --------------------------------------------------------------------------- #
# Optional input adapter: a reset linear first layer (new "visual" mapping)
# --------------------------------------------------------------------------- #
def _adapter_draw(population, adapter_dim, adapter_init_std, generator, device):
    """A random linear input adapter A (P, adapter_dim, 3). Default scale is
    variance-preserving for +/-1 inputs, so the latent keeps unit scale."""
    return (torch.randn(population, adapter_dim, 3, generator=generator,
                        device=device) * adapter_init_std)


def _shared_init_adapter(population, n_hidden, adapter_dim, weight_init_std,
                         small_init_std, use_output_bias, adapter_init_std,
                         generator, device, readout_init_std=None):
    """Shared random init for the adapter model: hidden ``W0`` reads from the
    ``adapter_dim`` latent (so its input dim is ``adapter_dim``, not 3), plus a
    random adapter ``A0`` mapping the 3 inputs into that latent. Hidden weights
    and hidden bias share ``weight_init_std``; the readout uses
    ``readout_init_std`` (falling back to ``small_init_std``)."""
    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device)
    if readout_init_std is None:
        readout_init_std = small_init_std
    return {
        "A0": _adapter_draw(population, adapter_dim, adapter_init_std, generator, device),
        "W0": randn(population, n_hidden, adapter_dim) * weight_init_std,
        "b0": randn(population, n_hidden) * weight_init_std,   # same layer as W0
        "v0": randn(population, n_hidden) * readout_init_std,
        "c0": (randn(population) * readout_init_std if use_output_bias
               else torch.zeros(population, device=device)),
    }


def _train_transfer_adapter(task, init, transfer_noise, num_steps, checkpoints,
                            conditions, learning_rate, optimizer_name,
                            use_output_bias, accuracy_mode, exception_prob,
                            generator, output_temperature=1.0):
    """Adapter-model analog of ``train_blob_grid`` for the type-5 transfer phase:
    the same paired standard/curriculum batched training and clean-prototype eval,
    but the forward is ``x -> A -> W(ReLU) -> v`` with a per-net linear adapter
    ``A``. Single noise level (K=1). Returns the same keys ``train_blob_grid``
    does (metrics of shape (C,1,T) with ``_sem``; ``final_*_per_seed`` (C,1,S)),
    so ``_shape_curve`` consumes it unchanged. Used only when ``adapter=True``;
    the shared kernel is untouched."""
    device = task.device
    conditions = list(conditions)
    C = len(conditions)
    P = init["W0"].shape[0]
    K, S = 1, P
    checkpoints = sorted({0} | {int(c) for c in checkpoints if 0 <= int(c) <= num_steps})
    checkpoint_set = set(checkpoints)
    has_adapter = "A0" in init

    def leaf(t, ndim):                                    # broadcast per-net -> (C, ...)
        return (t.to(device).unsqueeze(0).expand(C, *([-1] * ndim))
                .clone().requires_grad_(True))
    W = leaf(init["W0"], 3)                               # (C,P,H,a)
    b = leaf(init["b0"], 2)                               # (C,P,H)
    v = leaf(init["v0"], 2)                               # (C,P,H)
    params = [W, b, v]
    A = leaf(init["A0"], 3) if has_adapter else None      # (C,P,a,3)
    if A is not None:
        params.append(A)
    if use_output_bias:
        c = leaf(init["c0"], 1)                           # (C,P)
        params.append(c)
    else:
        c = torch.zeros(C, P, device=device)
    optimizer = build_optimizer(optimizer_name, params, learning_rate)

    p_sched = torch.stack([_condition_schedule(n, exception_prob, num_steps, device)
                           for n in conditions], 0)        # (C, T)
    protos = task.all_prototypes
    labels = task.all_labels
    is_periph = task.is_peripheral
    n_central = task.central_prototypes.shape[0]
    n_periph = task.peripheral_prototypes.shape[0]
    design = torch.cat([torch.ones(8, 1, device=device), protos], 1)   # (8,4)
    design_pinv = torch.linalg.pinv(design)
    targets = labels.view(1, 1, -1)

    metric_keys = ("peripheral_accuracy", "central_accuracy", "overall_accuracy",
                   "peripheral_loss", "central_loss", "overall_loss",
                   "participation_ratio", "active_unit_fraction", "logit_linearity")
    metrics = {k: [] for k in metric_keys}
    metrics_sem = {k: [] for k in metric_keys}
    final_ps = {}

    def evaluate():
        with torch.no_grad():
            if has_adapter:
                lat = torch.einsum("nd,cpad->cpna", protos, A)         # (C,P,8,a)
            else:
                lat = protos.view(1, 1, 8, -1).expand(C, P, -1, -1)
            pre = torch.einsum("cpna,cpha->cphn", lat, W) + b.unsqueeze(-1)  # (C,P,H,8)
            hidden = torch.relu(pre)
            logits = (torch.einsum("cphn,cph->cpn", hidden, v)
                      + c.unsqueeze(-1)) / output_temperature        # (C,P,8)
            if accuracy_mode == "expected":
                probs = torch.sigmoid(logits)
                correct = labels * probs + (1 - labels) * (1 - probs)
            else:
                correct = ((logits > 0).float() == labels).float()
            bce = F.binary_cross_entropy_with_logits(
                logits, targets.expand_as(logits), reduction="none")
            per_net = {
                "peripheral_accuracy": correct[:, :, is_periph].mean(-1),
                "central_accuracy": correct[:, :, ~is_periph].mean(-1),
                "overall_accuracy": correct.mean(-1),
                "peripheral_loss": bce[:, :, is_periph].mean(-1),
                "central_loss": bce[:, :, ~is_periph].mean(-1),
                "overall_loss": bce.mean(-1),
            }
            reps = hidden.permute(0, 1, 3, 2)
            reps = reps - reps.mean(dim=2, keepdim=True)
            lam = torch.linalg.svdvals(reps) ** 2
            per_net["participation_ratio"] = (
                lam.sum(-1) ** 2 / lam.pow(2).sum(-1).clamp_min(1e-12))
            per_net["active_unit_fraction"] = (
                (hidden > 0).any(dim=3).float().mean(dim=2))
            beta = torch.einsum("ab,cpb->cpa", design_pinv, logits)
            pred_lin = torch.einsum("sb,cpb->cps", design, beta)
            ss_res = ((logits - pred_lin) ** 2).sum(-1)
            ss_tot = ((logits - logits.mean(-1, keepdim=True)) ** 2).sum(-1).clamp_min(1e-8)
            per_net["logit_linearity"] = (1 - ss_res / ss_tot).clamp(-1, 1)
            for key, arr in per_net.items():
                grouped = arr.reshape(C, K, S)
                metrics[key].append(grouped.mean(-1).cpu().numpy())
                metrics_sem[key].append(
                    (grouped.std(-1, unbiased=False) / (S ** 0.5)).cpu().numpy())
            final_ps["peripheral"] = per_net["peripheral_accuracy"].reshape(C, K, S).cpu().numpy()
            final_ps["central"] = per_net["central_accuracy"].reshape(C, K, S).cpu().numpy()

    if 0 in checkpoint_set:
        evaluate()
    for t in range(num_steps):
        u = torch.rand(P, generator=generator, device=device)
        rule_idx = torch.randint(0, n_central, (P,), generator=generator, device=device)
        exc_idx = torch.randint(0, n_periph, (P,), generator=generator, device=device)
        noise = torch.randn(P, 3, generator=generator, device=device) * transfer_noise
        is_exc = u.unsqueeze(0) < p_sched[:, t].unsqueeze(1)          # (C,P)
        rule_x = task.central_prototypes[rule_idx]
        exc_x = task.peripheral_prototypes[exc_idx]
        base = torch.where(is_exc.unsqueeze(-1),
                           exc_x.unsqueeze(0), rule_x.unsqueeze(0))   # (C,P,3)
        x = base + noise.unsqueeze(0)
        y = torch.where(is_exc, task.peripheral_labels[exc_idx].unsqueeze(0),
                        task.central_labels[rule_idx].unsqueeze(0))
        optimizer.zero_grad()
        latent = torch.einsum("cpd,cpad->cpa", x, A) if has_adapter else x
        pre = torch.einsum("cpa,cpha->cph", latent, W) + b
        logit = ((torch.relu(pre) * v).sum(-1) + c) / output_temperature
        F.binary_cross_entropy_with_logits(logit, y, reduction="sum").backward()
        optimizer.step()
        if (t + 1) in checkpoint_set:
            evaluate()

    out = dict(noise_levels=np.array([transfer_noise]),
               checkpoints=np.array(checkpoints), conditions=list(conditions),
               n_hidden=int(W.shape[2]))
    for key in metrics:
        out[key] = np.stack(metrics[key], axis=-1)         # (C,1,T)
        out[key + "_sem"] = np.stack(metrics_sem[key], axis=-1)
    out["final_peripheral_per_seed"] = final_ps["peripheral"]   # (C,1,S)
    out["final_central_per_seed"] = final_ps["central"]
    return out


def train_pretrain_transfer(config, task, transfer_noise=0.8, pretrain_noise=None,
                            pretrain_steps=5000, transfer_steps=15000,
                            num_seeds=None, pretrain_kinds=("none", "1d", "xor"),
                            reset_readout=False, adapter=False, adapter_dim=3,
                            adapter_init_std=None, checkpoints=None,
                            pretrain_checkpoints=None, seed=0):
    """Full Experiment-2 run: shared init -> pretrain 1d/xor -> type-5 transfer.

    Model hyper-parameters (width, init scale, lr, optimizer, output bias, acc
    mode, exception prob) all come from ``config`` (a ``BlobConfig``); only the
    pretraining/transfer step budgets and noises are separate knobs.

    ``reset_readout=True`` carries over only the pretrained *hidden* weights
    (``W0, b0`` -- the learned features) and re-initialises the readout (``v0``
    and, if used, ``c0``) from a fresh small-random draw, isolating whether the
    compressed *representation* alone drives the curriculum effect. ``False``
    (default) is full weight carry-over.

    ``adapter=True`` inserts a linear input adapter as the first layer
    (``x -> A -> W(ReLU) -> v``) that is **reset (re-randomised) at the transfer
    boundary**: pretraining learns *through* an adapter, but the transfer phase
    gets a fresh one, so only the deeper representation (``W``, readout) carries
    over -- modelling humans who meet *new visual stimuli* on the second task
    while (possibly) reusing abstract structure. ``adapter_dim`` is the latent
    width (default 3), ``adapter_init_std`` its scale (default variance-
    preserving, ``1/sqrt(3)``). Runs a dedicated adapter training path; the
    shared kernel is used only when ``adapter=False``.

    Returns a dict with, per pretrain kind, a curve dict shaped like
    ``compute_curve_data`` (``<kind>`` -> ``{cond}_{group}[_sem]``), plus
    ``benefit_per_seed`` (curriculum - standard final exception accuracy) and
    ``pretrain_pr`` (per-seed pretrain participation ratio) for the correlation.
    """
    device = task.device
    S = int(config.num_seeds if num_seeds is None else num_seeds)
    H = config.n_hidden
    pretrain_noise = transfer_noise if pretrain_noise is None else pretrain_noise
    pretrain_kinds = list(pretrain_kinds)
    if checkpoints is None:
        checkpoints = _default_checkpoints(transfer_steps)
    if adapter_init_std is None:
        adapter_init_std = 1.0 / (3 ** 0.5)                 # variance-preserving
    _readout_std = getattr(config, "readout_init_std", None)
    g = torch.Generator(device=device).manual_seed(seed)

    # (1) one shared random init -- also the 'none' (from-scratch) baseline. With
    #     an adapter, W reads from the adapter_dim latent and a random A0 is added.
    if adapter:
        shared = _shared_init_adapter(S, H, adapter_dim, config.weight_init_std,
                                      config.small_init_std, config.use_output_bias,
                                      adapter_init_std, g, device,
                                      readout_init_std=_readout_std)
    else:
        shared = random_init_state(S, H, config.weight_init_std,
                                   config.small_init_std, config.use_output_bias,
                                   g, device, readout_init_std=_readout_std)

    def fresh_readout():
        f = random_init_state(S, H, config.weight_init_std, config.small_init_std,
                              config.use_output_bias, g, device,
                              readout_init_std=_readout_std)
        return f["v0"], f["c0"]

    # (2) pretrain the low-dimensional priors, (3) assemble phase-2 inits
    pre_results, inits = {}, {}
    for kind in pretrain_kinds:
        if kind == "none":
            inits["none"] = shared
            continue
        res = pretrain_population(
            task, kind, shared, pretrain_steps, pretrain_noise,
            config.learning_rate, config.optimizer, config.use_output_bias, g,
            checkpoints=pretrain_checkpoints, accuracy_mode=config.accuracy_mode,
            output_temperature=getattr(config, "output_temperature", 1.0))
        pre_results[kind] = res
        state = dict(res["state"])                          # W0,b0,v0,c0 (no adapter)
        if reset_readout:                                   # keep features, fresh readout
            v0, c0 = fresh_readout()
            state = {"W0": state["W0"], "b0": state["b0"], "v0": v0, "c0": c0}
        if adapter:                                         # reset adapter: fresh A0
            state["A0"] = _adapter_draw(S, adapter_dim, adapter_init_std, g, device)
        inits[kind] = state

    # (4) transfer each kind on the type-5 task, standard vs curriculum (paired
    #     init within a kind; a shared stimulus stream across kinds via seed+1)
    model_kw = dict(n_hidden=H, learning_rate=config.learning_rate,
                    weight_init_std=config.weight_init_std,
                    small_init_std=config.small_init_std,
                    readout_init_std=_readout_std,
                    optimizer_name=config.optimizer,
                    use_output_bias=config.use_output_bias,
                    accuracy_mode=config.accuracy_mode,
                    exception_prob=config.exception_prob,
                    output_temperature=getattr(config, "output_temperature", 1.0))
    result = dict(pretrain_kinds=pretrain_kinds,
                  conditions=["standard", "curriculum"],
                  transfer_noise=transfer_noise, pretrain_noise=pretrain_noise,
                  pretrain_steps=pretrain_steps, transfer_steps=transfer_steps,
                  n_hidden=H, num_seeds=S, weight_init_std=config.weight_init_std,
                  exception_prob=config.exception_prob, reset_readout=reset_readout,
                  adapter=adapter, adapter_dim=(adapter_dim if adapter else None),
                  accuracy_mode=config.accuracy_mode,
                  output_temperature=getattr(config, "output_temperature", 1.0),
                  benefit_per_seed={}, pretrain_pr={}, pretrain_accuracy={},
                  overlaps_rule={}, pretrain_curve={})
    for kind in pretrain_kinds:
        if adapter:
            grid = _train_transfer_adapter(
                task, inits[kind], transfer_noise, transfer_steps, checkpoints,
                ("standard", "curriculum"), config.learning_rate, config.optimizer,
                config.use_output_bias, config.accuracy_mode, config.exception_prob,
                generator=torch.Generator(device=device).manual_seed(seed + 1),
                output_temperature=getattr(config, "output_temperature", 1.0))
        else:
            grid = train_blob_grid(
                task=task, noise_levels=[transfer_noise], num_steps=transfer_steps,
                checkpoints=checkpoints, num_seeds=S,
                conditions=("standard", "curriculum"), init_state=inits[kind],
                return_final_per_seed=True, seed=seed + 1, **model_kw)
        curve, final_periph = _shape_curve(grid)           # final_periph (C, S)
        result[kind] = curve
        result["checkpoints"] = curve["checkpoints"]
        result["benefit_per_seed"][kind] = final_periph[1] - final_periph[0]  # (S,)
        if kind in pre_results:
            result["pretrain_pr"][kind] = pre_results[kind]["pretrain_pr"]
            result["pretrain_accuracy"][kind] = pre_results[kind]["pretrain_accuracy"]
            result["overlaps_rule"][kind] = pre_results[kind]["overlaps_rule"]
            result["pretrain_curve"][kind] = pre_results[kind]["curve"]
    return result


# --------------------------------------------------------------------------- #
# Plotting + summary
# --------------------------------------------------------------------------- #
def _band(ax, x, mean, sem, color, label=None):
    ax.plot(x, mean, "-", color=color, lw=2.4, label=label)
    if sem is not None:
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)


# metric -> colour (light blue exceptions, purple rule); pretrain kind -> line style
_METRIC_STYLE = (("peripheral", "exception accuracy", "#5aa9e6"),
                 ("central", "rule accuracy", "#8e44ad"))
_KIND_LINESTYLE = {"none": ":", "1d": "-", "xor": "--"}
_KIND_MARKER = {"none": "s", "1d": "o", "xor": "^"}
_COND_TITLE = {"standard": "no curriculum", "curriculum": "curriculum",
               "anti_curriculum": "anti-curriculum"}


def _acc_prefix(result):
    """'expected ' when accuracy is the soft/sigmoid metric, else '' (hard/argmax)."""
    return "expected " if result.get("accuracy_mode") == "expected" else ""


def _run_tag(result, brackets="()"):
    """Short run-modifier tag for titles, e.g. '  (adapter reset, readout reset)'."""
    bits = []
    if result.get("adapter"):
        bits.append("adapter reset")
    if result.get("reset_readout"):
        bits.append("readout reset")
    lo, hi = brackets
    return f"  {lo}{', '.join(bits)}{hi}" if bits else ""


def plot_transfer_by_schedule(result, kinds=None, figsize=(11, 4.4)):
    """Two panels (left = no curriculum, right = curriculum). In each, exception
    accuracy (light blue) and rule accuracy (purple) over transfer steps, with
    **line style** encoding the pretraining prior (1-D solid, XOR dashed, none
    dotted). Colour = what is measured, style = what it was pretrained on."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    format_axis = _try_format_axis()
    conds = result["conditions"]
    if kinds is None:
        kinds = [k for k in result["pretrain_kinds"] if k in result]
    x = np.asarray(result["checkpoints"])

    fig, axes = plt.subplots(1, len(conds), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    data_lines = []
    for ax, cond in zip(axes, conds):
        for kind in kinds:
            curve = result[kind]
            ls = _KIND_LINESTYLE.get(kind, "-")
            for key, _, color in _METRIC_STYLE:
                mean = curve[f"{cond}_{key}"]
                (ln,) = ax.plot(x, mean, ls=ls, color=color)
                data_lines.append(ln)
                sem = curve.get(f"{cond}_{key}_sem")
                if sem is not None:
                    ax.fill_between(x, mean - sem, mean + sem, color=color,
                                    alpha=0.12, lw=0)
        ax.axhline(0.5, ls=":", lw=1.0, color="0.6", zorder=0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("transfer step")
        ax.set_title(_COND_TITLE.get(cond, cond))
        format_axis(ax, line_width_multiplier=1, font_size_multiplier=0.6)
    for ln in data_lines:                                   # re-thicken after format_axis
        ln.set_linewidth(2.3)
    pre = _acc_prefix(result)
    axes[0].set_ylabel(f"{pre}accuracy")

    # two keys: colour = measured group, line style = pretraining prior
    colour_handles = [Line2D([0], [0], color=c, lw=2.6, label=f"{pre}{lab}")
                      for _, lab, c in _METRIC_STYLE]
    style_handles = [Line2D([0], [0], color="0.3", lw=2.3,
                            ls=_KIND_LINESTYLE.get(k, "-"),
                            label=f"pretrain: {_KIND_LABEL.get(k, k)}") for k in kinds]
    leg1 = axes[0].legend(handles=colour_handles, frameon=False, fontsize=9,
                          loc="lower left")
    axes[0].add_artist(leg1)
    axes[-1].legend(handles=style_handles, frameon=False, fontsize=9,
                    loc="lower left")
    reset = _run_tag(result)
    fig.suptitle(
        f"Transfer accuracy by schedule   (noise {result['transfer_noise']}, "
        f"{result['num_seeds']} seeds, width {result['n_hidden']}){reset}",
        y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def plot_final_accuracy_by_schedule(result, kinds=None, ylim=None,
                                    figsize=(6.8, 4.2)):
    """Human-style final-accuracy dot plot: curriculum condition on the x-axis,
    final centroid accuracy on the y-axis, **colour = centroid type** (light blue
    exceptions, purple rule) and **marker = pretraining prior** (1-D circle, XOR
    triangle, none square). Points are dodged within each condition and carry a
    +/-1 SEM (over seeds) error bar; dashed line at chance."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    format_axis = _try_format_axis()
    conds = result["conditions"]
    if kinds is None:
        kinds = [k for k in result["pretrain_kinds"] if k in result]
    pre = _acc_prefix(result)

    # dodge: cluster by centroid type (all exceptions together, all rule together)
    # within each curriculum, then spread the pretraining priors inside a cluster
    metrics = list(_METRIC_STYLE)
    n_metric, n_kind = len(metrics), len(kinds)
    group_span, within_span = 0.34, 0.16
    gcen = (np.linspace(-group_span / 2, group_span / 2, n_metric)
            if n_metric > 1 else np.array([0.0]))
    wcen = (np.linspace(-within_span / 2, within_span / 2, n_kind)
            if n_kind > 1 else np.array([0.0]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.5, ls="--", lw=1.3, color="black", zorder=0)
    lo, hi = 1.0, 0.0
    for xc, cond in enumerate(conds):
        for mi, (key, _lab, color) in enumerate(metrics):
            for ki, kind in enumerate(kinds):
                off = gcen[mi] + wcen[ki]
                c = result[kind]
                mean = float(c[f"{cond}_{key}"][-1])
                sem = c.get(f"{cond}_{key}_sem")
                sem = float(sem[-1]) if sem is not None else 0.0
                ax.errorbar(xc + off, mean, yerr=sem,
                            fmt=_KIND_MARKER.get(kind, "o"), color=color, ms=8,
                            mec="white", mew=0.8, elinewidth=1.6, capsize=3, zorder=3)
                lo, hi = min(lo, mean - sem), max(hi, mean + sem)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([_COND_TITLE.get(cd, cd) for cd in conds])
    ax.set_xlim(-0.5, len(conds) - 0.5)
    if ylim is None:
        ylim = (min(0.45, lo - 0.03), min(1.02, hi + 0.03))
    ax.set_ylim(*ylim)
    ax.set_ylabel(f"final {pre}accuracy")
    ax.set_xlabel("curriculum")

    colour_handles = [Line2D([0], [0], marker="o", ls="", color=col, ms=9,
                             mec="white", label=f"{pre}{lab}")
                      for _key, lab, col in _METRIC_STYLE]
    shape_handles = [Line2D([0], [0], marker=_KIND_MARKER.get(k, "o"), ls="",
                            color="0.3", ms=9, mec="white",
                            label=_KIND_LABEL.get(k, k)) for k in kinds]
    leg1 = ax.legend(handles=colour_handles, frameon=False, fontsize=9,
                     loc="upper left", bbox_to_anchor=(1.02, 1.0), title="centroid")
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.02, 0.48), title="pretraining")
    format_axis(ax, line_width_multiplier=1, font_size_multiplier=0.7)
    reset = _run_tag(result)
    ax.set_title(f"Final accuracy  (noise {result['transfer_noise']}, "
                 f"{result['num_seeds']} seeds){reset}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.72, 1))                  # reserve right margin for legends
    return fig


def plot_pretrain_transfer(result, figsize=None):
    """Figure-4 analog: rows = {exception accuracy, participation ratio}, columns
    = pretrain kinds {none, 1d, xor}; standard vs curriculum overlaid (mean +/-1
    SEM over seeds) across transfer steps."""
    import matplotlib.pyplot as plt
    format_axis = _try_format_axis()
    kinds = [k for k in result["pretrain_kinds"] if k in result]
    conds = result["conditions"]
    x = np.asarray(result["checkpoints"])
    pre = _acc_prefix(result)
    rows = (("peripheral", f"{pre}exception accuracy", (0.0, 1.02), 0.5),
            ("central", f"{pre}rule accuracy", (0.0, 1.02), 0.5),
            ("participation_ratio", "participation ratio", None, None))
    nrows, ncols = len(rows), len(kinds)
    if figsize is None:
        figsize = (3.9 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for r, (key, ylabel, ylim, ref) in enumerate(rows):
        for cix, kind in enumerate(kinds):
            ax = axes[r][cix]
            curve = result[kind]
            for cond in conds:
                _band(ax, x, curve[f"{cond}_{key}"], curve.get(f"{cond}_{key}_sem"),
                      _COND_COLOR.get(cond, None), label=_COND_LABEL.get(cond, cond))
            if ref is not None:
                ax.axhline(ref, ls=":", lw=1.2, color="0.5", zorder=0)
            if key == "participation_ratio":
                for lv in (1, 2, 3):
                    ax.axhline(lv, ls=":", lw=0.8, color="0.8", zorder=0)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(_KIND_LABEL.get(kind, kind))
            if r == nrows - 1:
                ax.set_xlabel("transfer step")
            if cix == 0:
                ax.set_ylabel(ylabel)
            if r == 0 and cix == 0:
                ax.legend(frameon=False, fontsize=9, loc="lower right")
            format_axis(ax, line_width_multiplier=1, font_size_multiplier=0.6)
    reset = _run_tag(result)
    fig.suptitle(
        f"Pretraining -> type-5 transfer   "
        f"(noise {result['transfer_noise']}, {result['num_seeds']} seeds,"
        f" width {result['n_hidden']}){reset}", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def plot_pretrain_curves(result, figsize=(13, 3.8)):
    """The **pretraining phase** itself over training: clean-corner accuracy, BCE
    loss and participation ratio (mean +/-1 SEM over seeds), 1-D and XOR priors
    overlaid. Shows the compressed 1-D prior settling to a lower PR than XOR --
    the compression that drives the later curriculum benefit."""
    import matplotlib.pyplot as plt
    format_axis = _try_format_axis()
    curves = result.get("pretrain_curve", {})
    kinds = [k for k in result["pretrain_kinds"] if k in curves]
    if not kinds:
        raise ValueError("no pretrain_curve recorded (only 'none' kind?)")
    panels = (("accuracy", f"{_acc_prefix(result)}clean-corner accuracy", (0.0, 1.02), 0.5),
              ("loss", "BCE loss", None, None),
              ("participation_ratio", "participation ratio", None, None))
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (key, ylabel, ylim, ref) in zip(axes, panels):
        for kind in kinds:
            cv = curves[kind]
            _band(ax, cv["checkpoints"], cv[key], cv.get(key + "_sem"),
                  _KIND_COLOR.get(kind), label=_KIND_LABEL.get(kind, kind))
        if ref is not None:
            ax.axhline(ref, ls=":", lw=1.2, color="0.5", zorder=0)
        if key == "participation_ratio":
            for lv in (1, 2, 3):
                ax.axhline(lv, ls=":", lw=0.8, color="0.8", zorder=0)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("pretraining step")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9)
        format_axis(ax, line_width_multiplier=1, font_size_multiplier=0.6)
    fig.suptitle(
        f"Pretraining phase  (noise {result['pretrain_noise']}, "
        f"{result['num_seeds']} seeds, width {result['n_hidden']}, "
        f"{result['pretrain_steps']} steps)", y=1.03, fontsize=12)
    fig.tight_layout()
    return fig


def plot_compression_predicts_benefit(result, kind="1d", n_bins=6, figsize=(5.2, 4.4)):
    """Fig-5C analog: per-seed pretrain participation ratio (x) vs per-seed
    curriculum benefit on the exceptions (y), with quantile-binned means. More
    compressed prior (lower PR) -> larger curriculum benefit is a negative
    slope."""
    import matplotlib.pyplot as plt
    format_axis = _try_format_axis()
    if kind not in result.get("pretrain_pr", {}):
        raise ValueError(f"no pretrain PR recorded for kind {kind!r} "
                         f"(have {list(result.get('pretrain_pr', {}))})")
    pr = np.asarray(result["pretrain_pr"][kind])
    benefit = np.asarray(result["benefit_per_seed"][kind])
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, ls=":", lw=1.0, color="0.6", zorder=0)
    ax.scatter(pr, benefit, s=16, alpha=0.4, color=_COND_COLOR.get("curriculum"),
               edgecolors="none", zorder=2)
    # quantile bins -> mean +/- SEM
    order = np.argsort(pr)
    edges = np.quantile(pr, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bx, by, be = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pr >= lo) & (pr < hi)
        if m.sum() >= 2:
            bx.append(pr[m].mean()); by.append(benefit[m].mean())
            be.append(benefit[m].std(ddof=0) / np.sqrt(m.sum()))
    ax.errorbar(bx, by, yerr=be, fmt="-o", color="k", lw=2.2, ms=6,
                capsize=3, zorder=3, label="binned mean")
    if pr.std() > 1e-9:
        r = float(np.corrcoef(pr, benefit)[0, 1])
        ax.set_title(f"{_KIND_LABEL.get(kind, kind)} prior:  r = {r:+.2f}")
    ax.set_xlabel("pretrain participation ratio (compression)")
    ax.set_ylabel("curriculum benefit on exceptions")
    ax.legend(frameon=False, fontsize=9)
    format_axis(ax, line_width_multiplier=1, font_size_multiplier=0.7)
    fig.tight_layout()
    return fig


def pretrain_summary(result):
    """Print final accuracies, start/final PR and the curriculum benefit for each
    (pretrain kind x schedule)."""
    kinds = [k for k in result["pretrain_kinds"] if k in result]
    conds = result["conditions"]
    reset = _run_tag(result, brackets="[]")
    print(f"Pretraining -> type-5 transfer  |  noise {result['transfer_noise']}, "
          f"{result['num_seeds']} seeds, width {result['n_hidden']}, "
          f"pretrain {result['pretrain_steps']} / transfer {result['transfer_steps']} "
          f"steps{reset}")
    header = (f"{'prior':>12} {'sched':>11} | {'excep':>6} {'rule':>6} "
              f"{'PR0':>5} {'PRf':>5}")
    print(header); print("-" * len(header))
    for kind in kinds:
        curve = result[kind]
        pr_txt = ""
        if kind in result.get("pretrain_pr", {}):
            pr_txt = f"  <pretrain PR {np.mean(result['pretrain_pr'][kind]):.2f}>"
        for cond in conds:
            exc = curve[f"{cond}_peripheral"][-1]
            rule = curve[f"{cond}_central"][-1]
            pr0 = curve[f"{cond}_participation_ratio"][0]
            prf = curve[f"{cond}_participation_ratio"][-1]
            print(f"{_KIND_LABEL.get(kind, kind):>12} {cond:>11} | "
                  f"{exc:6.3f} {rule:6.3f} {pr0:5.2f} {prf:5.2f}")
        ben = float(np.mean(result["benefit_per_seed"][kind]))
        sem = float(np.std(result["benefit_per_seed"][kind], ddof=0)
                    / np.sqrt(len(result["benefit_per_seed"][kind])))
        print(f"{'':>12} {'curr-benefit':>11} | {ben:+6.3f} +/- {sem:.3f}"
              f" on exceptions{pr_txt}")
        print()
