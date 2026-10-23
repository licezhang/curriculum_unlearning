# Methods notes — decisions that must reach the paper

Working record of modelling and analysis choices that are load-bearing, non-obvious, or
easy to get wrong when the Methods section is written. Everything here reflects the
**current** implementation (`paper_lib.py` + `paper_figures.ipynb`). Anything describing the
older non-yoked model has been removed; treat pre-September results as superseded.

Last updated 2026-10-06.

---

## 1. Experiment 1 model

**Architecture.** Two-layer feedforward network: 3 inputs → `H` ReLU hidden units → 1 sigmoid
output. Inputs are the Boolean stimulus in {−1,1}³.

**Noise is on the hidden activations, not the input.** `h = relu(Wx + b) + ε`, with
ε ~ N(0, σ²) and σ = 1.0, drawn fresh on every trial. There is no input noise in the
selected model. Earlier drafts described "Gaussian noise around each stimulus prototype" —
that is wrong for this model and must be corrected.

**Yoking.** One network per participant (n = 207), trained by single-trial SGD on that
participant's exact 240-trial sequence in their curriculum condition, with feedback after
every trial. Networks are therefore matched to humans trial for trial, and the model has no
freedom over its own training distribution.

**Lapse.** A single group-level lapse ε = 0.30, applied as an observation model
(`p' = (1−ε)p + ε/2`) to responses only — never to the weight update. Use a *group* lapse for
any reported contrast; see §5.

**Selected configuration:** 32 hidden units, lr 0.30, init scale 0.010, hidden noise 1.0,
lapse 0.30. In the MLE notebook this configuration is labelled `rich`.

### 1a. Model selection — resolved, with one correction

Two configurations appear in `yoked_mle_results.ipynb`:

| | lr | init | hidden noise | selected by |
|---|---|---|---|---|
| `fitted` | 0.10 | 0.20 | 0.0 | best held-out likelihood on human responses |
| `rich` | 0.30 | 0.01 | 1.0 | best fit to the group learning curves |

The paper uses `rich`. An earlier version of this note claimed that the init *scale* selects
rich vs lazy and that the best-fitting model was therefore the one least able to show
uncompression. **That was wrong**, and the test that settles it (2026-10-06) is:

| | S→C − NC in final PR | |
|---|---|---|
| init 0.20, hn 0.0 | −0.010 (p = .74) | the `fitted` model |
| init 0.20, hn 1.0 | +0.116 (p < .001) | |
| init 0.10, hn 0.0 | −0.035 | |
| init 0.10, hn 1.0 | +0.135 (p < .001) | trajectory winner |
| init 0.01, hn 0.0 | +0.015 (p = .32) | |
| init 0.01, hn 1.0 | +0.104 (p < .001) | paper model |

The uncompression contrast is absent at **every** initialisation scale when hidden noise is
zero and present at every scale when it is one. It is the hidden noise that matters, not the
init scale. `fitted` fails to show uncompression because its noise is 0.

This removes the apparent conflict between fit and mechanism, because hidden noise is
**unidentifiable under the per-participant-lapse likelihood** (§1c) — the lapse absorbs it, so
every noisy generator is fitted with noise = 0. The held-out-likelihood criterion cannot speak
to the parameter the mechanism needs. The trajectory criterion, which uses a single group
lapse, can, and prefers noise near 1.0: every configuration in the top six has hidden noise
between 0.75 and 1.5. There is no "we chose the model that agrees" problem to disclose.

### 1b. Hyperparameter sweep

Script: `reconstruct_sweep.py` (writes `results/sweep_exp1.csv`).

**The initialisation scale is fixed at 0.01 and is not swept.** This is a theoretical
commitment: the account concerns networks trained from small weights, and the init scale is
the parameter that selects rich vs lazy. It is also the parameter the per-participant-lapse
likelihood cannot identify (§1c), so fixing it loses nothing estimable.

**Swept:** learning rate [0.05, 0.10, 0.20, 0.30, 0.50] x hidden-activation noise
[0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50] = 35 configs at 32 hidden units, three seeds each.

**Criterion:** RMSE against the human group learning curves over the 56 defined
block x condition x stimulus-type cells (10 bins x 3 conditions x 2 stimulus types, less the
4 the design leaves undefined), minimising over a single global lapse on a 31-point grid from
0 to 0.6.

**Result:** lr 0.30, hidden noise 1.0 wins, at RMSE 0.0770 +/- 0.0011 over 15 seeds. Quote
0.077 in the text.

**What is and is not identified.** The learning rate is sharply identified: the runner-up,
0.20, is worse by 0.0035 (paired t(14) = 18.80, p < .001), and 0.10 and 0.05 are far worse
(0.110, 0.172). The hidden noise is not: 0.75, 1.25 and 1.50 are statistically
indistinguishable from 1.0 (p = .19, .49, .07). Write "learning rate 0.30 with hidden noise
in 0.75-1.5, of which 1.0 was selected", not "noise of 1.0 was identified". What matters for
the mechanism is that noise is non-zero: noise = 0 gives RMSE 0.101 and abolishes the
uncompression contrast entirely (§1a).

The reconstruction recovers a fitted lapse of 0.28, the grid point adjacent to the 0.30 the
model uses; the two are indistinguishable at the grid's 0.02 resolution.

**Provenance caveat.** The original driver was lost. Reproducing the criterion, the 56-cell
count and the selected configuration is good evidence the procedure is right, but these ranges
should be reported as what the released script sweeps, not as verified historical endpoints.
`--sweep-init` restores the earlier 175-config version, in which the paper model ranks 5/175
(0.0769 vs 0.0758 for lr 0.20 / init 0.100 / noise 1.0, paired t(9) = -3.17, p = .011) --
expected, since the init scale there is being fitted rather than fixed by theory.

### 1c. Validation results that constrain claims

From the MLE analysis (`yoked_mle_results.ipynb`):

- **Seed noise is small.** Per-participant log-lik SD across seeds 0.016/trial, against a
  between-config spread of 0.078. 5–10 seeds suffice.
- **Only the learning rate is recoverable.** Hidden noise is completely unidentifiable (the
  per-participant lapse absorbs it); init recovery is at chance. Do not interpret
  per-participant parameter estimates.
- **Per-participant architecture fitting overfits.** +0.022/trial in sample, −0.023 on the
  second half, −0.014 on held-out test. Fit the architecture globally, the lapse per
  participant.

---

## 2. Measurement conventions

**Learning curves** come from the training forward pass, with hidden noise injected — i.e.
measured as humans are, trial by trial on the stimulus actually presented.

**Test accuracy** marginalises the hidden noise over 400 draws. This matters: computing
curves with noise and test without inflated test exceptions by 0.103 against rules by 0.035,
which is a real bug that was fixed.

**Binning is by presented trial number, not by rank among surviving trials.** A participant
who lost a trial to a timeout has every later stimulus shift one rank earlier, which leaks
block-2 exception trials into a rank-based bin 2 and puts a stray point on the
simple-to-complex exception curve where the design guarantees none.

**Participation ratio** PR = (Σλ)²/Σλ² on the centred post-ReLU representations of the eight
stimuli, computed on the **noise-free** code. Accuracy quantities marginalise noise; PR and
the task axes do not.

**Task axes.** Rule axis = class-1 minus class-0 mean over the six rule stimuli. Exception
axis = the direction separating the two exceptions, orthogonalised against the rule axis.
Both have a fixed sign in every network, so projections can be averaged across networks.

---

## 3. Strategy classifier

Ten candidates per participant-block: nine deterministic policies plus random responding.
The nine are the correct rule; rule plus exceptions; the correct feature with inverted sign;
each of the two irrelevant features in either sign; and the two constant responses. These
collapse to five reported labels (five incorrect 1-D rules → "other 1-D rule"; two constants
→ "constant").

Each policy is fitted with a maximum-likelihood lapse (the disagreement rate, clipped to
[1e-6, 0.5]) and compared by BIC with a penalty of log(N) for that one parameter. **The random
null is unpenalised**, so classification defaults to random unless a structured policy earns
its parameter — the "random" category is deliberately conservative.

**Identifiability.** A block is scored only if the stimuli actually presented distinguish
`rule` from `rule + exceptions`. This is why blocks are blank in the figures (simple-to-complex
block 1, complex-to-simple block 5).

**Power caveat.** The test bar rests on 16 trials (two presentations of each stimulus) against
48 per learning block. Subsampling a learning block to 16 trials drops a significant
interaction to 48% detection, so the test column is the least-powered bar in the panel.

---

## 4. Experiment 2 model

**Yoking through both phases.** One network per participant (n = 1027), 240 pretraining +
240 R+E trials, in that participant's own **physical** feature coding.

**Feature coding — resolved empirically.** `stimuli` in the raw file is LOGICAL coding shared
by both phases. `stimulusRandomization` holds two entries, `red` and `blue`, with no column
saying which phase is which. Resolved against `simple_rule_match`: **red = pretraining,
blue = the R+E phase (and test), and `featureAssignment` maps logical → physical.** That
combination reproduces `simple_rule_match` for 100% of the 525 Simple-pretrained participants;
the three alternatives manage 59–65%. Pretraining rule = logical bit 0; R+E rule = logical
bit 1. Whether `flipFeatureValues` indexes logical or physical position is not determinable
from the repo, but it only decides which input dimension carries a sign flip, which random
symmetric init makes immaterial in aggregate.

**Hyperparameters inherited unchanged from Experiment 1.** Only two lapse rates are fitted,
0.10 for pretraining and 0.24 for the R+E phase.

**No weight-scale pullback (α = 0).** A partial renormalisation at the phase boundary was
explored and rejected: across five seeds it does not affect the key interaction (flat in α,
well inside the seed spread), the fit optimum sits at α = 0.20 rather than 0.10, and the
pretraining effect — which humans show at −0.099 — decays as α rises and reverses by α = 0.30.
α = 0.1 is a compromise, not an optimum. Kept as a supplementary figure.

**Known limitation.** The networks have almost no between-network variance: PR after
pretraining has a within-cell SD of 0.082 against a manipulation effect of 1.841 (4%), and
within-cell SD of exception accuracy is 0.091 against 0.329 in humans. Yoking supplies matched
trial sequences but not individual differences, so the human individual-differences analysis
has no model counterpart. Reproducing it would need per-network heterogeneity (varying
learning rate, noise, capacity or lapse).

---

## 5. Statistics

**Language split is deliberate.** Python owns everything involving the networks and every
descriptive/figure quantity; R owns the mixed models on human trial-level data. Both should
emit into `results/` with a `source` field so the two halves merge into one report.

**Estimator matters, and Python has no trustworthy GLMM.** Checked against the published
human three-way interaction (+0.77, p = .020):

| estimator | 3-way | verdict |
|---|---|---|
| GLMMadaptive (R) | +0.769, p = .024 | reproduces to 3 d.p. — **use this** |
| statsmodels `BinomialBayesMixedGLM` | +0.728, p < .001 | right estimate, z inflated ~3× |
| cluster-robust GLM | +0.356, p = .133 | marginal coefficient — flips the verdict |

`lme4` will not build on this machine (`nloptr` fails to configure); `GLMMadaptive` installs
cleanly and matches. Script: `r_scripts/experiment2_threeway.R`.

**Continuous vs binary network outputs answer different questions.** Sampling 16 binary
responses per network and fitting the human GLMM asks "would an experiment with this many
networks detect the effect" — the answer is no. Using the continuous P(correct), as the
earlier modelling did, asks whether the effect is present in the model — the answer is yes
(three-way +0.052, t = 4.14, p < .001). Report the continuous form; say which you used.

**The per-participant lapse must not be used for group contrasts.** It is fitted to each
participant's own responses, differs by condition (NC 0.31, S→C 0.12, C→S 0.35; F = 23.0,
p = 1e-9) and correlates r = −0.66 with their test accuracy. The S→C − NC rule contrast flips
from −0.021 (raw network) to +0.072 with it, against a human +0.061 — the human effect
arriving through the nuisance parameter. Use the per-participant lapse for likelihood only;
use a single group lapse for anything reported.

**Human–network correspondence.** Two permutation tests, and they disagree:
- *Condition level* (main text): human and network strategy distributions over blocks differ
  by 0.091 for matched curricula against 0.149 mismatched; permuting network condition labels
  gives 0.120, p < .001.
- *Individual level* (supplement): a network agrees with its own participant on 36.6% of
  blocks, which is **not** better than a random partner in the same condition (null 37.5%,
  p = .78), though better than a random partner overall (31.8%, p < .001). Yoking buys
  condition-level correspondence, not individual-level.

---

## 6. Claims from the old model that do NOT survive

- **"Simplicity bias persists after 30,000 training steps."** The yoked model trains for
  exactly 240 trials. Separately, when trained to convergence the curriculum benefit is
  neutral-to-harmful — the benefit is specific to a compressed prior combined with a limited
  training budget. The long-training claim points the other way from the Introduction.
- **"Curriculum benefit is largest for low-dimensional initialisations" (old Fig 4C).**
  Shaping the singular-value spectrum of W at fixed Frobenius norm was tested and had no
  effect on anything: ∂L/∂W = δxᵀ has full support over input dimensions whatever W's rank,
  so a discarded direction is refilled on the next trial. What remains is correlational —
  final PR predicts final exception accuracy across networks at r = +0.70.
- **Complex-to-simple "catastrophic forgetting below chance" in networks.** The yoked model
  falls to chance (0.492); humans fall below it (0.396).
- **Exceptions are not "overwritten" under complex-to-simple.** At the end of training the
  exception-axis separation is 3.04 against 4.25 and 5.12 for the other curricula — reduced
  but present — while the *logit* separation is −0.24 against +10.90 and +12.80. The
  representation survives; the readout re-aligns onto the rule axis and stops reading it.

---

## 7. Stale artefacts to retire before packaging

- `results/main_sweep.pkl`, `hidden_sweep.pkl`, `init_scale_sweep.pkl`, `metadata.json`,
  `nn_curriculum_summary.pkl` — all from the old non-yoked model (`num_steps`,
  `exception_prob`, `peripheral`/`central` naming). Move to `archive/` or delete.
- `r_scripts/experiment1_network_stats.Rmd`, `experiment2_network_stats.Rmd` read
  `net_exp*.csv` exports from the old model.
- `paper_figures_PRE_RESHUFFLE_2026-10-02.ipynb` / `paper_lib_PRE_RESHUFFLE_2026-10-02.py`
  are the pre-reshuffle backups; keep until the new figure layout is settled.

## Exception frequency confounds cross-curriculum reaction-time contrasts

The exception RT cost (mean log-RT, exception minus rule, per participant and block) is only
comparable across blocks within the **no curriculum** condition, where exceptions are 25 % of
RT-eligible trials in every block (25.0 / 25.0 / 25.0 / 25.0 / 25.0). In the curriculum
conditions, exception share *is* the manipulation:

| condition | b1 | b2 | b3 | b4 | b5 |
|---|---|---|---|---|---|
| Complex to Simple | 50.0 % | 37.6 % | 25.0 % | 12.5 % | — |
| No Curriculum | 25.0 % | 25.0 % | 25.0 % | 25.0 % | 25.0 % |
| Simple to Complex | — | 12.5 % | 25.0 % | 37.5 % | 50.0 % |

Rarer stimuli are responded to more slowly, so a block-pooled cross-curriculum contrast
confounds strategy with exposure frequency, and the confound predicts the same ordering as the
strategy account (S→C sees exceptions least often on average and shows the largest cost; C→S
sees them most often and shows none). It also explains why the C→S cost is significantly
*negative* in block 2 (−5.8 %, p < .001), where exceptions are over-represented at 37.6 %.

Consequences for the text:
- The developmental claim is made **within no curriculum**, where measurement is constant:
  null in blocks 1–2 (−0.3 %, p = .88; +1.5 %, p = .39), reliable from block 3 (+7.8 %,
  p < .001), slope +0.0295 log-units/block, t(62) = 3.90, p < .001.
- Cross-curriculum claims use **block 3 only**, where all three conditions sit at 25 %:
  S→C +6.9 points, p = .046; C→S −5.2 points, p = .10. Matched, the effect is marginal for
  S→C and absent for C→S.
- `fig4.rt_cost_by_curriculum.*` (block-pooled) is retained in the report but flagged
  CONFOUNDED. Do not cite it.
- Trends in the curriculum conditions (`fig4.rt_cost_trend.{curriculum,anti_curriculum}`) are
  likewise uninterpretable as learning: C→S rises (+0.0254, p = .015) while its exception
  share falls, and S→C is flat (p = .47) while its share rises.

## Why the network test models are OLS, not a GLM

This was a deliberate choice in the code (`exp2_threeway(..., ols=True)`, with a binomial
branch behind the flag) but the reason was never written down. Recorded here.

1. **There is no binomial denominator.** The outcome is each network's P(correct),
   marginalised over 400 hidden-noise draws — an expectation, not *k* successes out of *n*
   trials. The 400 draws are our measurement resolution, not anything the network
   experienced, so a binomial family would be fitting a bare proportion with an arbitrary
   implied weight.
2. **The logit link buys nothing over the observed range.** Network accuracies span
   0.343–0.880, with none outside [0.1, 0.9], so logit is near-linear throughout.
3. **The estimand is a probability difference.** The text reports Δ in accuracy units and
   the 75/25 occurrence weighting operates on probabilities; OLS gives these directly, a
   logit model gives log-odds needing back-transformation.

**Robustness check (2026-10-12).** Refitting the Experiment 2 test model with
`family=Binomial()`, cluster-robust by network, changes nothing:

| | three-way | on exceptions |
|---|---|---|
| OLS (reported) | b = +0.052, t = 4.14, p = 3.5e-5 | b = +0.033, t = 2.85, p = .0044 |
| binomial GLM | b = +0.315, t = 4.37, p = 1.2e-5 | b = +0.174, t = 2.80, p = .0051 |

Coefficients are on different scales (probability vs log-odds); compare t and p.

**Do not confuse this with the §5 warning** that a cluster-robust GLM returns marginal
coefficients. That concerned replacing a GLMM on *human trial-level* data, where the random
effects carry real variance. For the networks there is no random-effect structure to
preserve — the random-intercept variance is estimated as exactly zero, because a network's
rule and exception accuracy are negatively correlated — so a marginal coefficient is the
target anyway.

**Terminology.** These are linear models on per-network proportions. Do not call them GLMs
in the manuscript: it invites the question of which family and link, and the answer is
neither.
