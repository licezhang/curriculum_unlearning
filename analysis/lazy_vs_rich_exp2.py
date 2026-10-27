"""Can a network that cannot reshape its representation produce the Exp. 2 interaction?

Experiment 2's three-way interaction -- pretraining x curriculum x stimulus type -- is the
one result in the paper that the recency confound cannot touch, because both pretraining
groups face identical R+E schedules. It is therefore the load-bearing test, and it is worth
showing that it requires the mechanism we claim.

The lesion is the same as in lazy_vs_rich.py: freeze the hidden layer at initialisation and
train the readout only. That network still carries something across the phase boundary --
its readout weights -- so pretraining can still affect it. What it cannot do is arrive at
the R+E phase with a representation that 1-D pretraining compressed and XOR pretraining did
not, because its representation never changes. Its participation ratio after pretraining is
identical in the two groups by construction.

If the three-way interaction survives the lesion, readout carryover is enough and our
mechanism is not needed. If it does not, Exp. 2 is evidence for the mechanism specifically.

    python lazy_vs_rich_exp2.py                # 3 seeds -> results/lazy_vs_rich_exp2.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_lib as P

ARMS = (('rich', dict()), ('lazy', dict(init=1.0, freeze=True)))


def contrasts(m):
    """-> dict of the quantities the paper reports for Exp. 2."""
    out = {}
    out['threeway'] = P.lmm_contrast(m, {'pre01:cur01:exc_c': 1})
    for pre, lbl in ((1, 'oneD'), (0, 'xor')):
        # curriculum effect on EXCEPTION stimuli within one pretraining group
        out[f'exc_benefit_{lbl}'] = P.lmm_contrast(m, {
            'cur01': 1, 'cur01:exc_c': .5, 'pre01:cur01': pre, 'pre01:cur01:exc_c': .5 * pre})
        out[f'rule_benefit_{lbl}'] = P.lmm_contrast(m, {
            'cur01': 1, 'cur01:exc_c': -.5, 'pre01:cur01': pre, 'pre01:cur01:exc_c': -.5 * pre})
    out['exc_difference'] = P.lmm_contrast(m, {'pre01:cur01': 1, 'pre01:cur01:exc_c': .5})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='results/lazy_vs_rich_exp2.csv')
    a = ap.parse_args()

    D2 = P.YokedExp2Data()
    print(f'{D2.P} yoked networks, Exp. 2 (pretraining -> R+E -> test)', flush=True)

    rows, cells, prs = [], [], []
    for arm, cfg in ARMS:
        tot, pr_by, pr_all = 0.0, [], []
        for s in range(a.seeds):
            R2 = P.run_yoked_exp2(D2, cfg=cfg, seed=s)
            tot = tot + R2['pc_test']
            pr_by.append({p: R2['pr'][D2.pre == p].mean() for p in ('Simple', 'XOR')})
            pr_all.append({'all': R2['pr']})
            print(f'  {arm} seed {s} done', flush=True)
        R2m = {'pc_test': tot / a.seeds, 'cfg': {**P.YOKED_EXP2, **cfg}}
        df = P.test_long_exp2(D2, R2m)
        m = P.test_lmm(df, 'p ~ pre01 * cur01 * exc_c')
        C = contrasts(m)
        # per-network cell means, for the figure's accuracy panels
        cl = (df.groupby(['pre', 'cur', 'is_exc', 'net']).p.mean().reset_index()
              .groupby(['pre', 'cur', 'is_exc']).p
              .agg(mean='mean', sd=lambda x: x.std(ddof=1), n='size').reset_index())
        cl['sem'] = cl.sd / np.sqrt(cl.n); cl['arm'] = arm
        cells.append(cl)
        # participation ratio after pretraining, per network
        prr = pd.DataFrame(dict(arm=arm, pre=D2.pre.astype(str),
                                pr=np.mean([x['all'] for x in pr_all], 0)))
        prs.append(prr.groupby(['arm', 'pre']).pr
                   .agg(mean='mean', sd=lambda x: x.std(ddof=1), n='size').reset_index())
        prS = np.mean([x['Simple'] for x in pr_by]); prX = np.mean([x['XOR'] for x in pr_by])
        print(f'\n{arm}: PR after pretraining  1-D {prS:.3f}   XOR {prX:.3f}   '
              f'(difference {prS - prX:+.3f})')
        for k, (b, lo, hi, z, p) in C.items():
            print(f'  {k:<20} {b:+.4f} [{lo:+.4f}, {hi:+.4f}]  z = {z:+.2f}  p = {p:.3g}')
            rows.append(dict(arm=arm, quantity=k, est=b, lo=lo, hi=hi, z=z, p=p,
                             pr_oneD=prS, pr_xor=prX, seeds=a.seeds))

    B = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    B.to_csv(a.out, index=False)
    pd.concat(cells).to_csv(a.out.replace('.csv', '_cells.csv'), index=False)
    pd.concat(prs).to_csv(a.out.replace('.csv', '_pr.csv'), index=False)
    print(f'\nwrote {a.out} ({len(B)} rows), plus _cells.csv and _pr.csv')


if __name__ == '__main__':
    main()
