"""The Experiment 1 hyperparameter sweep.

The initialisation scale is FIXED at 0.01. This is a theoretical commitment, not a fitting
outcome: the account concerns networks trained from small weights (the rich regime), and the
init scale is the parameter that selects rich vs lazy. Fixing it also removes a parameter
that the per-participant-lapse likelihood cannot identify at all. What is swept is the
learning rate and the hidden-activation noise -- 35 configs at 32 hidden units.

Each config is scored by RMSE against the human group learning curves over the 56 defined
block x condition x stimulus-type cells (10 bins x 3 conditions x 2 stimulus types, less the
4 the design leaves undefined), minimising over a single global lapse.

Selected: lr 0.30, hidden noise 1.0 (RMSE 0.0770 +/- 0.0011 over 15 seeds). The learning rate
is sharply identified -- the runner-up, 0.20, is worse by 0.0035, paired t(14) = 18.8 --
whereas noise levels 0.75 to 1.5 are statistically indistinguishable from 1.0. Noise of ZERO
is catastrophic (RMSE 0.101) and also abolishes the uncompression signature, so what the data
support is "non-zero noise in 0.75-1.5", not "noise is 1.0".

    python reconstruct_sweep.py                 # the 35-config sweep -> results/sweep_exp1.csv
    python reconstruct_sweep.py --sweep-init    # also sweep init (175 configs; see METHODS_NOTES 1b)
    python reconstruct_sweep.py --quick         # coarse smoke test
"""
import argparse, itertools, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F

sys.path.insert(0, '.')
import paper_lib as P

LR_GRID = [0.05, 0.10, 0.20, 0.30, 0.50]
INIT_FIXED = 0.010                                     # rich regime, fixed by theory
INIT_GRID = [0.005, 0.010, 0.020, 0.050, 0.100]        # only with --sweep-init
HN_GRID = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
LAPSE_GRID = np.linspace(0.0, 0.6, 31)
NBIN, BIN_TRIALS, HIDDEN = 10, 24, 32


def human_matrix(HC):
    """(3, 10, 2) of human mean accuracy: condition x bin x (rule, exception)."""
    M = np.full((3, NBIN, 2), np.nan)
    for ci, c in enumerate(P.CONDS):
        for k, st in enumerate(('Rule', 'Exception')):
            q = HC[(HC.cond == c) & (HC.stimType == st)].set_index('bin')['mean']
            for b in range(1, NBIN + 1):
                if b in q.index:
                    M[ci, b - 1, k] = q.loc[b]
    return M


def train_pc(D, lr, init, hn, seed=0):
    """Per-trial P(correct). Only what the learning curves need -- no PR probes."""
    g = torch.Generator(device=D.dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, device=D.dev)
    W = (rn(D.P, HIDDEN, 3) * init).requires_grad_(True)
    b = (rn(D.P, HIDDEN) * init).requires_grad_(True)
    v = (rn(D.P, HIDDEN) * init).requires_grad_(True)
    par = [W, b, v]
    pc = np.zeros((D.P, D.T))
    for t in range(D.T):
        h = torch.relu(torch.einsum('pd,phd->ph', D.PRt[D.IDX[:, t]], W) + b)
        h = h + rn(D.P, HIDDEN) * hn
        lg = (h * v).sum(-1); y = D.LAB[:, t]
        with torch.no_grad():
            q = torch.sigmoid(lg)
            pc[:, t] = (y * q + (1 - y) * (1 - q)).cpu().numpy()
        loss = (F.binary_cross_entropy_with_logits(lg, y, reduction='none') * D.MSK[:, t]).sum()
        gs = torch.autograd.grad(loss, par)
        with torch.no_grad():
            for p_, gr in zip(par, gs):
                p_ -= lr * gr
    return pc


def net_matrix(D, pc):
    """(3, 10, 2), measured exactly as the humans are."""
    out = np.full((3, NBIN, 2), np.nan)
    edges = np.linspace(0, D.T, NBIN + 1).astype(int)
    for ci, c in enumerate(P.CONDS):
        m = (D.cond == c)
        for bi in range(NBIN):
            sl = slice(edges[bi], edges[bi + 1])
            for k in (0, 1):
                w = D.msk[m][:, sl] & (D.exc_trial[m][:, sl] == bool(k))
                n = w.sum(1)
                if (n > 0).any():
                    per = (pc[m][:, sl] * w).sum(1)[n > 0] / n[n > 0]
                    out[ci, bi, k] = per.mean()
    return out


def score(net, hum):
    """Best RMSE over the defined cells, minimising over a single global lapse."""
    ok = ~np.isnan(net) & ~np.isnan(hum)
    best = (np.inf, np.nan)
    for e in LAPSE_GRID:
        r = np.sqrt(np.mean(((1 - e) * net[ok] + e / 2 - hum[ok]) ** 2))
        if r < best[0]:
            best = (r, e)
    return best[0], best[1], int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--sweep-init', action='store_true',
                    help='also sweep the initialisation scale (175 configs)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='results/sweep_exp1.csv')
    ap.add_argument('--stability', type=int, default=0, metavar='N',
                    help='re-run the top 5 configs over N seeds -> results/sweep_exp1_stability.csv')
    a = ap.parse_args()
    inits = INIT_GRID if a.sweep_init else [INIT_FIXED]
    lrs, hns = (([0.1, 0.3], [0.0, 0.5, 1.0]) if a.quick else (LR_GRID, HN_GRID))
    if a.quick and a.sweep_init: inits = [0.01, 0.1]

    LRN, _T = P.load_exp1()
    D = P.YokedData(LRN)
    HUM = human_matrix(P.learning_curves_human(LRN))
    combos = list(itertools.product(lrs, inits, hns))
    print(f'{len(combos)} configs x {a.seeds} seeds, {D.P} networks, {D.T} trials each')
    rows, t0 = [], time.time()
    for i, (lr, init, hn) in enumerate(combos):
        nets = [net_matrix(D, train_pc(D, lr, init, hn, seed=s)) for s in range(a.seeds)]
        rmse, eps, ncell = score(np.nanmean(np.stack(nets), 0), HUM)
        rows.append(dict(lr=lr, init=init, hnoise=hn, rmse=rmse, lapse=eps, cells=ncell))
        if (i + 1) % 10 == 0 or i + 1 == len(combos):
            print(f'  {i + 1}/{len(combos)}  ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).sort_values('rmse').reset_index(drop=True)
    df.to_csv(a.out, index=False)
    print(f'\nwrote {a.out}  ({len(df)} configs, {rows[0]["cells"]} cells scored)\n')
    print(df.head(10).to_string(index=False))
    w = df.iloc[0]
    tgt = (abs(w.lr - .30) < 1e-9 and abs(w.init - .010) < 1e-9 and abs(w.hnoise - 1.0) < 1e-9)
    print(f'\nwinner: lr {w.lr}, init {w.init}, hidden noise {w.hnoise}, '
          f'lapse {w.lapse:.2f}, RMSE {w.rmse:.4f}')
    print('MATCHES the model in the paper' if tgt else
          'DOES NOT match the paper model (lr 0.30 / init 0.010 / noise 1.0)')
    if not tgt:
        m = df[(df.lr == .30) & (df.init == .010) & (df.hnoise == 1.0)]
        if len(m):
            print(f'  the paper model ranks {m.index[0] + 1}/{len(df)} at RMSE {m.iloc[0].rmse:.4f}')
            print('  (expected with --sweep-init: the init scale is fixed by theory, not fitted)')
    if a.stability:
        stability(D, HUM, df, a.stability)


def stability(D, HUM, df, n_seeds, top=5, out='results/sweep_exp1_stability.csv'):
    """Re-run the leading configs over more seeds. The 3-seed ranking is within seed
    noise at the top, so the reported numbers should come from here, not from the grid."""
    from scipy import stats as sstats
    sel = df.head(top)[['lr', 'init', 'hnoise']].values.tolist()
    runs = {}
    print(f'\nstability: top {top} configs x {n_seeds} seeds')
    for lr, init, hn in sel:
        v = np.array([score(net_matrix(D, train_pc(D, lr, init, hn, seed=s)), HUM)[0]
                      for s in range(n_seeds)])
        runs[(lr, init, hn)] = v
        print(f'  lr {lr:<5} init {init:<6} hn {hn:<5} RMSE {v.mean():.5f} +/- {v.std(ddof=1):.5f}')
    best = min(runs, key=lambda k: runs[k].mean())
    rows = []
    for k, v in runs.items():
        t = sstats.ttest_rel(v, runs[best]) if k != best else None
        rows.append(dict(lr=k[0], init=k[1], hnoise=k[2], seeds=n_seeds,
                         rmse=v.mean(), sd=v.std(ddof=1), is_best=(k == best),
                         delta_vs_best=v.mean() - runs[best].mean(),
                         t_vs_best=(t.statistic if t else np.nan),
                         p_vs_best=(t.pvalue if t else np.nan)))
    out_df = pd.DataFrame(rows).sort_values('rmse').reset_index(drop=True)
    out_df.to_csv(out, index=False)
    print(f'  wrote {out}')
    return out_df


if __name__ == '__main__':
    main()
