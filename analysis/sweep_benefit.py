"""The curriculum benefit across the Experiment 1 sweep grid (Figure 3e, f).

`reconstruct_sweep.py` scores each configuration by how well it reproduces the human
learning curves, but does not record the effect of interest. This script re-runs the same
grid and stores the curriculum benefit -- simple-to-complex minus no curriculum, in
end-of-training accuracy -- separately for exception and rule stimuli.

Every cell is a full yoked simulation: 207 networks, each trained on its own participant's
240-trial sequence in that participant's curriculum, re-run at that configuration. The grid
varies the model only; the data and the design are held fixed throughout.

The point of the figure is robustness. The exception benefit is positive in 34 of the 35
configurations -- the exception being zero hidden noise at the highest learning rate -- and
the configuration the paper uses (lr 0.30, sigma 1.0) is NOT the one that maximises it
(+0.068 against +0.103 at lr 0.20, sigma 1.5). So the effect does not depend on where in the
grid the fitted model happens to sit.

    python sweep_benefit.py              # 10 seeds per config -> results/sweep_exp1_benefit.csv
    python sweep_benefit.py --seeds 3    # fewer seeds, for a quick check

Takes about 20 minutes at 10 seeds on one GPU. Each configuration's own fitted lapse is read
from results/sweep_exp1.csv, so run reconstruct_sweep.py first.
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_lib as P

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--out', default='results/sweep_exp1_benefit.csv')
A = ap.parse_args()

SW = pd.read_csv('results/sweep_exp1.csv')
SW = SW[np.isclose(SW.init, 0.01)]                  # init is fixed by theory, not swept
LAPSE = {(r.lr, r.hnoise): r.lapse for r in SW.itertuples()}
LRS, NOISE = sorted(SW.lr.unique()), sorted(SW.hnoise.unique())

LRN, _ = P.load_exp1()
D = P.YokedData(LRN)
sc, nc = D.cond == 'curriculum', D.cond == 'standard'
print(f'{len(LRS)} learning rates x {len(NOISE)} noise levels x {A.seeds} seeds, '
      f'{D.P} yoked networks each', flush=True)

rows = []
for lr in LRS:
    for hn in NOISE:
        eps = LAPSE.get((lr, hn), 0.30)             # that configuration's own fitted lapse
        for sd in range(A.seeds):
            R = P.run_yoked(D, cfg=dict(lr=lr, hnoise=hn), seed=sd, pr_every=10 ** 9)
            raw = R['final']                        # noise-marginalised, pre-lapse
            for tag, a in (('raw', raw), ('lapsed', (1 - eps) * raw + eps / 2)):
                e, r_ = a[:, P.IS_EXC].mean(1), a[:, ~P.IS_EXC].mean(1)
                rows.append(dict(lr=lr, hnoise=hn, seed=sd, lapse=eps, scale=tag,
                                 exc_benefit=e[sc].mean() - e[nc].mean(),
                                 rule_benefit=r_[sc].mean() - r_[nc].mean(),
                                 exc_sc=e[sc].mean(), exc_nc=e[nc].mean()))
        print(f'  lr={lr:<5} sigma={hn:<5} done', flush=True)

B = pd.DataFrame(rows)
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
B.to_csv(A.out, index=False)
g = B[B.scale == 'lapsed'].groupby(['lr', 'hnoise']).exc_benefit
print(f'\nwrote {A.out} ({len(B)} rows)')
print(f'  exception benefit positive in {(g.mean() > 0).sum()} of {g.ngroups} configurations')
print(f'  seed SD: median {g.std().median():.4f}, max {g.std().max():.4f}')
