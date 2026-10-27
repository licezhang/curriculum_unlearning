"""Does the curriculum benefit survive a balanced washout block? (Tier 1b)

Both reviewers raise the same confound: the test phase follows block 5 immediately, and
block 5 is 50% exceptions under simple-to-complex, 25% under no curriculum and 0% under
complex-to-simple. The ordering of test exception accuracy is therefore what a
recency-weighted learner would predict, with no appeal to compression.

The experiment they ask for is a balanced consolidation block before test. We cannot run it
in people without new data collection, but we can run it in the networks, which is where the
recency account has to hold if it is going to explain the humans too.

Every network gets an identical extra block appended after block 5: 48 trials at the
no-curriculum composition (36 rule, 12 exception; each rule stimulus six times, each
exception six times), order shuffled per network. All three conditions then end on the same
local statistics, so anything that survives is not recency.

    python consolidation_control.py                 # 5 seeds -> results/consolidation.csv
    python consolidation_control.py --seeds 1       # quick
"""
import argparse, copy, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_lib as P

N_EXTRA = 48


def extend(D, n=N_EXTRA, seed=0):
    """D with an identical balanced block appended to every network's sequence."""
    rng = np.random.default_rng(seed)
    rule_idx, exc_idx = np.where(~P.IS_EXC)[0], np.where(P.IS_EXC)[0]
    reps = n // 8                                    # 6: each stimulus six times
    base = np.concatenate([np.repeat(rule_idx, reps), np.repeat(exc_idx, reps)])
    assert len(base) == n and P.IS_EXC[base].sum() == n // 4, 'block must be 25% exceptions'
    extra = np.stack([rng.permutation(base) for _ in range(D.P)])

    E = copy.copy(D)
    E.T = D.T + n
    E.idx = np.concatenate([D.idx, extra], 1)
    E.msk = np.concatenate([D.msk, np.ones((D.P, n), bool)], 1)
    E.exc_trial = P.IS_EXC[E.idx]
    t = lambda a, d=torch.float32: torch.tensor(a, device=D.dev, dtype=d)
    E.IDX = t(E.idx, torch.long)
    E.LAB = torch.cat([D.LAB, t(D.labels8[extra])], 1)
    E.MSK = t(E.msk.astype(np.float32))
    return E


def summarise(D, R, tag, seed):
    acc = R['final']                                 # noise-marginalised, pre-lapse
    eps = P.YOKED_EXP1['lapse']
    acc = (1 - eps) * acc + eps / 2
    e_, r_ = acc[:, P.IS_EXC].mean(1), acc[:, ~P.IS_EXC].mean(1)
    m = {c: D.cond == c for c in P.CONDS}
    out = dict(run=tag, seed=seed)
    for c in P.CONDS:
        out[f'exc_{c}'] = e_[m[c]].mean(); out[f'rule_{c}'] = r_[m[c]].mean()
    out['exc_benefit'] = out['exc_curriculum'] - out['exc_standard']
    out['rule_benefit'] = out['rule_curriculum'] - out['rule_standard']
    out['exc_cost_anti'] = out['exc_anti_curriculum'] - out['exc_standard']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='results/consolidation.csv')
    a = ap.parse_args()

    LRN, _ = P.load_exp1()
    D = P.YokedData(LRN)
    print(f'{D.P} networks, {D.T} trials; washout adds {N_EXTRA} balanced trials', flush=True)

    rows, acc_sum = [], {'baseline': 0.0, 'washout': 0.0}
    for s in range(a.seeds):
        for tag, DD in (('baseline', D), ('washout', extend(D, seed=s))):
            R = P.run_yoked(DD, seed=s, pr_every=10 ** 9)
            rows.append(summarise(DD, R, tag, s))
            acc_sum[tag] = acc_sum[tag] + R['final']
        print(f'  seed {s} done', flush=True)
    # the simulation seed is a nuisance of the model, not of the data: average it out
    # before fitting, so the model is not reading one draw of the hidden noise
    longs = []
    for tag in ('baseline', 'washout'):
        df = P.test_long_exp1(D, {'final': acc_sum[tag] / a.seeds})
        df['run'] = tag; longs.append(df)

    B = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    B.to_csv(a.out, index=False)
    g = B.groupby('run')[['exc_benefit', 'rule_benefit', 'exc_cost_anti',
                          'exc_curriculum', 'exc_standard', 'exc_anti_curriculum']]
    print(f'\nwrote {a.out} ({len(B)} rows)\n')
    print('mean over seeds (accuracy, lapsed):')
    print(g.mean().to_string())
    print('\nseed SD:')
    print(g.std().to_string())

    print(f'\nmixed model on accuracy averaged over {a.seeds} seeds, within stimulus type:')
    L = pd.concat(longs)

    # per-network cell means, for the figure's accuracy panels
    cells = (L.groupby(['run', 'cond', 'is_exc', 'net']).p.mean().reset_index()
             .groupby(['run', 'cond', 'is_exc']).p
             .agg(mean='mean', sd=lambda x: x.std(ddof=1), n='size').reset_index())
    cells['sem'] = cells.sd / np.sqrt(cells.n)
    cells.to_csv(a.out.replace('.csv', '_cells.csv'), index=False)

    crows = []
    for tag in ('baseline', 'washout'):
        m = P.test_lmm(L[L.run == tag], 'p ~ (sc + cs) * exc_c')
        for nm, w in (('exception', .5), ('rule', -.5)):
            for cc, lbl in (('sc', 'S->C'), ('cs', 'C->S')):
                est, lo, hi, z, pv = P.lmm_contrast(m, {cc: 1, f'{cc}:exc_c': w})
                crows.append(dict(run=tag, contrast=lbl, stim=nm, est=est, lo=lo, hi=hi,
                                  z=z, p=pv))
                print(f'  {tag:<9} {lbl:<5} {nm:<10} {est:+.4f} '
                      f'[{lo:+.4f}, {hi:+.4f}]  z = {z:+.2f}  p = {pv:.3g}')
    pd.DataFrame(crows).to_csv(a.out.replace('.csv', '_contrasts.csv'), index=False)
    print(f"\nalso wrote {a.out.replace('.csv', '_cells.csv')} and "
          f"{a.out.replace('.csv', '_contrasts.csv')}")


if __name__ == '__main__':
    main()
