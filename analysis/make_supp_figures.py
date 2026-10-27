"""The two supplementary figures for the revision.

S1  the Exp. 1 balanced-washout control      <- results/consolidation*.csv
S2  the Exp. 2 frozen-representation control <- results/lazy_vs_rich_exp2*.csv

Both read the frozen outputs of the analysis scripts, so this draws and does not compute.
Run `consolidation_control.py --seeds 20` and `lazy_vs_rich_exp2.py --seeds 3` first.

    python make_supp_figures.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_lib as P

P.use_style()
# run from the repository root (as run.py arranges) the figures belong in figures/;
# run from a flat working directory they go next to the script
OUT = 'figures' if os.path.isdir('figures') else '.'
RUN_LBL = {'baseline': 'No washout', 'washout': 'Balanced washout'}
ARM_LBL = {'rich': 'Trained', 'lazy': 'Frozen'}
PRE_C = {'Simple': P.INK, 'XOR': P.MUT}
PRE_LBL = {'Simple': 'One-dimensional', 'XOR': 'XOR'}
MK = {'Simple': 'o', 'XOR': 's'}
DX = 0.17


def pts(ax, x, y, err, color, filled, label=None, mk='o', ms=4.4):
    """One marker with a 95% interval, open for the reference level and filled for the test."""
    ax.errorbar(x, y, yerr=err, fmt=mk, ms=ms, color=color, mfc=color if filled else 'white',
                mew=1.3, lw=0, elinewidth=1.1, capsize=2.1, ecolor=color, label=label,
                zorder=3, clip_on=False)


# ------------------------------------------------------------------ Figure S1 ----
C = pd.read_csv('results/consolidation_cells.csv')
K = pd.read_csv('results/consolidation_contrasts.csv')

fig = plt.figure(figsize=(7.2, 2.25))
gs = fig.add_gridspec(1, 3, left=.075, right=.995, bottom=.235, top=.845, wspace=.46,
                      width_ratios=[1, 1, 1.5])
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# a, b -- test accuracy by stimulus type, each condition before and after the washout
for k, (ax, is_exc, nm) in enumerate(zip(axes[:2], (True, False), ('Exception', 'Rule'))):
    for xi, c in enumerate(P.CONDS):
        for run, filled in (('baseline', False), ('washout', True)):
            r = C[(C.run == run) & (C.cond == c) & (C.is_exc == is_exc)].iloc[0]
            pts(ax, xi + (DX if filled else -DX), r['mean'], 1.96 * r['sem'],
                P.condColors[c], filled)
    ax.axhline(.5, color=P.MUT, lw=P.LW_REF, ls=(0, (2.4, 1.8)), zorder=1)
    ax.set(xlim=(-.5, 2.5), xticks=range(3),
           xticklabels=[P.condShort[c] for c in P.CONDS], ylim=(.45, .9),
           yticks=np.arange(.5, .91, .1))
    ax.set_ylabel('Test accuracy' if k == 0 else None)
    P.finish(ax, f'{nm} stimuli')
    if k == 1:
        # neutral proxy handles: the open/filled contrast is about the run, not the condition
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker='o', ms=4.4, lw=0, color=P.MUT, mew=1.3,
                    mfc='white' if j == 0 else P.MUT) for j in (0, 1)]
        ax.legend(h, [RUN_LBL['baseline'], RUN_LBL['washout']], frameon=False,
                  loc='center', bbox_to_anchor=(.5, .32), handletextpad=.4, borderpad=0,
                  labelspacing=.3)
        P.finish(ax)

# c -- the same contrasts, before and after
ax = axes[2]
order = [('S->C', 'exception'), ('C->S', 'exception'), ('S->C', 'rule'), ('C->S', 'rule')]
CC = {'S->C': P.condColors['curriculum'], 'C->S': P.condColors['anti_curriculum']}
for xi, (cc, st) in enumerate(order):
    for run, filled in (('baseline', False), ('washout', True)):
        r = K[(K.run == run) & (K.contrast == cc) & (K.stim == st)].iloc[0]
        pts(ax, xi + (DX if filled else -DX), r.est, [[r.est - r.lo], [r.hi - r.est]],
            CC[cc], filled)
ax.axhline(0, color=P.INK, lw=P.LW_REF, zorder=1)
ARROW = {'S->C': 'S→C', 'C->S': 'C→S'}
ax.set(xlim=(-.5, 3.5), xticks=range(4),
       xticklabels=[f'{ARROW[c]}\n{"exc." if s == "exception" else "rule"}'
                    for c, s in order],
       ylim=(-.145, .085), yticks=np.arange(-.12, .081, .04))
ax.set_ylabel('Contrast vs no curriculum')
P.finish(ax, 'Curriculum effects at test')

for ax, s in zip(axes, 'abc'): P.panel_letter(fig, ax, s)
for ext in ('svg', 'png'):
    fig.savefig(os.path.join(OUT, f'figureS1_washout.{ext}'), dpi=400)
plt.close(fig)
print(f'wrote {OUT}/figureS1_washout.svg / .png')


# ------------------------------------------------------------------ Figure S2 ----
PR = pd.read_csv('results/lazy_vs_rich_exp2_pr.csv')
B = pd.read_csv('results/lazy_vs_rich_exp2.csv')

fig = plt.figure(figsize=(7.2, 2.25))
gs = fig.add_gridspec(1, 3, left=.075, right=.995, bottom=.235, top=.845, wspace=.46,
                      width_ratios=[1, 1.25, 1])
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

ax = axes[0]
for xi, arm in enumerate(('rich', 'lazy')):
    for pre, off in (('Simple', -DX), ('XOR', DX)):
        r = PR[(PR.arm == arm) & (PR.pre == pre)].iloc[0]
        pts(ax, xi + off, r['mean'], 1.96 * r.sd / np.sqrt(r.n), PRE_C[pre], True,
            mk=MK[pre])
ax.set(xlim=(-.5, 1.5), xticks=[0, 1], xticklabels=[ARM_LBL[a] for a in ('rich', 'lazy')],
       ylim=(.8, 3.7), yticks=(1, 2, 3))
ax.set_ylabel('Participation ratio')
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], marker=MK[k], ms=4.4, lw=0, color=PRE_C[k]) for k in PRE_C],
          [PRE_LBL[k] for k in PRE_C], frameon=False, loc='center left',
          bbox_to_anchor=(-.03, .44), handletextpad=.35, borderpad=0, labelspacing=.3)
P.finish(ax, 'After pretraining')

ax = axes[1]
for xi, arm in enumerate(('rich', 'lazy')):
    for pre, key, off in (('Simple', 'exc_benefit_oneD', -DX), ('XOR', 'exc_benefit_xor', DX)):
        r = B[(B.arm == arm) & (B.quantity == key)].iloc[0]
        pts(ax, xi + off, r.est, [[r.est - r.lo], [r.hi - r.est]], PRE_C[pre], True,
            mk=MK[pre])
ax.axhline(0, color=P.INK, lw=P.LW_REF, zorder=1)
ax.set(xlim=(-.5, 1.5), xticks=[0, 1], xticklabels=[ARM_LBL[a] for a in ('rich', 'lazy')],
       ylim=(-.01, .16), yticks=np.arange(0, .161, .05))
ax.set_ylabel('Curriculum benefit\n(exception stimuli)')
P.finish(ax, 'Benefit by pretraining')

ax = axes[2]
for xi, arm in enumerate(('rich', 'lazy')):
    r = B[(B.arm == arm) & (B.quantity == 'threeway')].iloc[0]
    pts(ax, xi, r.est, [[r.est - r.lo], [r.hi - r.est]], P.INK, True, ms=5.0)
ax.axhline(0, color=P.INK, lw=P.LW_REF, zorder=1)
ax.set(xlim=(-.5, 1.5), xticks=[0, 1], xticklabels=[ARM_LBL[a] for a in ('rich', 'lazy')],
       ylim=(-.035, .075), yticks=np.arange(-.02, .071, .02))
ax.set_ylabel('Estimate')
P.finish(ax, 'Three-way interaction')

for ax, s in zip(axes, 'abc'): P.panel_letter(fig, ax, s)
for ext in ('svg', 'png'):
    fig.savefig(os.path.join(OUT, f'figureS2_exp2_lesion.{ext}'), dpi=400)
plt.close(fig)
print(f'wrote {OUT}/figureS2_exp2_lesion.svg / .png')
