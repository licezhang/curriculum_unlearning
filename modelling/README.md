# Gaussian-blob inputs and curriculum benefit

Does a rule-first **curriculum** help more when the type-5 rule+exception inputs
are **Gaussian blobs** (`signal_noise_std > 0`) than when they are the 8 fixed
examples (`std = 0`)? And how does that interact with **training length** and
**hidden width**?

Network: an `n_hidden`-unit ReLU MLP (readout + output bias) from **small random
weights**, trained online (batch size 1) with BCE. Each step's input is a type-5
prototype (one of 6 rule / 2 exception corners) plus fresh Gaussian jitter of std
`signal_noise_std` on the 3 coordinates. `std = 0` recovers the 8 fixed examples.
Three exposure schedules run in parallel, **matched for mean exposure** and
differing only in timing: **standard** holds the exception probability at
`exception_prob`; **curriculum** (simple→complex) ramps it `0 → 2·exception_prob`;
**anti_curriculum** (complex→simple) ramps `2·exception_prob → 0` (the exact
mirror). Plots show all three (curriculum should help exceptions, anti-curriculum
should hurt them via catastrophic forgetting).

**Evaluation** — accuracy *and* BCE loss are measured on the **clean 8
prototypes**, i.e. the blob *centres*, not on noisy samples. So the metrics ask
"where is the decision boundary at each cloud centre", not generalisation over
the cloud. Losses are reported split by group: rule (6 central), exception (2
peripheral), and combined (all 8). Curves also carry a **±1 SEM band over seeds**.

**Mechanism metrics** (logged per checkpoint, plotted over training in the
notebook, std vs curr with SEM): **participation ratio** -- the effective
dimensionality (~1-7) of the 8 centroids' *centered, post-ReLU* hidden
representation (tests whether curriculum keeps the representation *uncompressed*
rather than collapsing onto the rule axis) -- and **active-unit fraction** (share
of hidden units firing on >=1 centroid; the recruitment partner to PR).

**Accuracy mode** (`accuracy_mode`, `--accuracy-mode`): `argmax` (default) is hard
0/1 accuracy from thresholding the logit; `expected` is the soft accuracy = the
probability the sigmoid assigns to the *true* label (so confidence counts, and a
barely-correct prediction scores ~0.5). Only the accuracy metrics change; the BCE
loss is identical. In the notebook it's the `ACCURACY_MODE` knob in the Controls
cell.

## Layout

| file | purpose |
|------|---------|
| `blob_task.py` | Signed type-5 prototypes (6 central / 2 peripheral) + labels. |
| `blob_training.py` | `train_blob_grid(...)` — batched over (condition × noise × seed) for a fixed width/step budget; checkpoint eval on clean prototypes. |
| `blob_experiments.py` | `BlobConfig` + `main_sweep` (noise × steps at fixed width), `hidden_sweep` (width × noise), and `init_scale_sweep` (hidden-init scale × noise; rich vs lazy). |
| `blob_plotting.py` | Benefit-vs-noise, (noise × steps) heatmaps for exception *and* rule accuracy, (noise × width) and (init-scale × noise) heatmaps, learning + loss curves. |
| `run_blob_experiments.py` | Runner CLI → pickles into `results/`. |
| `blob_analysis.ipynb` | Load-and-plot (plus one optional live-query cell). |

## Usage

```bash
cd scripts/notebooks/gaussian_blob_curriculum
python run_blob_experiments.py                 # full run (auto GPU, ~30s)
python run_blob_experiments.py --quick         # tiny smoke test
python run_blob_experiments.py --optimizer adam --lr 0.02 --n-hidden 32
python run_blob_experiments.py --only main_sweep
```

Env: the `simplicitybias-gpu` conda env.

## What it reuses / how it differs from `two_neuron_phase`

Same task and the same paired, vectorised-over-seeds training style. It differs
in the model (N hidden ReLU units from small random init, not 2 angle-placed
units) and the inputs (Gaussian blobs, sampled per step), which is exactly the
axis this study varies. The whole (condition × noise × seed) population trains in
one tensor, so a sweep is seconds.

## Result (default full run)

Curriculum benefit on the exceptions is **~0 at `std = 0`** (both conditions
solve the fixed 8) and **grows as inputs become blobs**, peaking around
`std ≈ 1.25–1.5` (benefit ≈ +0.25–0.27) before declining as the task gets too
noisy for either condition. Mechanistically, moderate input noise is what lets
standard training slide onto the rule-only solution (its exception accuracy falls
from ~1.0 to ~0.5 as noise grows) while the curriculum still carves the pocket.

Width interacts mainly at the **low end**: very narrow nets (2–4 units) can be
*hurt* by curriculum at low noise, whereas width ≥ 8 shows ~0 benefit at low
noise and a strong positive benefit at moderate noise, with a similar peak
(~0.27). See `blob_analysis.ipynb` for the figures.

**Init scale (rich vs lazy).** The `init_scale_sweep` varies `weight_init_std`
(small = rich/feature-learning, large = lazy). The regime strongly modulates the
curriculum:
- Small init (rich, e.g. 0.05–0.1): curriculum *hurts* at low noise (benefit
  down to ~−0.15) but gives the *largest* benefit at high noise (~+0.36).
- Large init (lazy, e.g. 1–2): ~0 benefit at low noise, and the high-noise
  benefit shrinks / collapses (curriculum's own exception accuracy degrades).

So the default `weight_init_std = 0.5` sits on the lazy-ish side (zero low-noise
benefit); pushing toward the rich regime makes curriculum both more harmful at
low noise and more helpful at high noise. At high noise the nets also **trade
rule accuracy for exception accuracy** — visible as a blue region in the rule
(central) benefit heatmap.
