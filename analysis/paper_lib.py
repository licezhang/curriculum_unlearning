"""Analyses and style for the main paper figures.

Everything the figure notebook needs lives here: the plotting contract, the data
loaders, and each analysis as a function returning tidy frames. Nothing in here
draws a figure, and nothing is hard-coded to one experiment where it need not be.
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib as mpl, matplotlib.pyplot as plt

# ---------------------------------------------------------------- style -----
FS_TICK, FS_LABEL, FS_TITLE = 7, 8, 9
LW_DATA, LW_REF, LW_SPINE, LEN_TICK = 1.6, 0.9, 0.8, 2.5
INK, MUT = '#1a1723', '#6f6979'

# stimulus type -- inherited from Figure 1, do not repurpose
stimColors = {'Rule': '#8e7cc3', 'Exception': '#6fa8dc'}
# curriculum condition
condColors = {'standard': '#7f7f7f', 'curriculum': '#1b9e77', 'anti_curriculum': '#d95f02'}
condLabel = {'standard': 'No Curriculum', 'curriculum': 'Simple to Complex',
             'anti_curriculum': 'Complex to Simple'}
HUMAN_COND = {'standard': 'No\nCurriculum', 'curriculum': 'Simple\nTo Complex',
              'anti_curriculum': 'Complex\nTo Simple'}
CONDS = ['anti_curriculum', 'standard', 'curriculum']   # C->S, NC, S->C
condShort = {'anti_curriculum': 'C→S', 'standard': 'NC', 'curriculum': 'S→C'}
# strategy state -- ordered by sophistication, deliberately clear of Figure 1's cool pair
STATES = ['random', 'constant', 'other 1-D rule', 'rule', 'rule + exceptions']
SHORT = ['random', 'constant', 'other 1-D', 'rule', 'rule + exc.']
stateColors = {'random': '#dcd9e0', 'constant': '#b0aab9', 'other 1-D rule': '#847d92',
               'rule': '#8e7cc3', 'rule + exceptions': '#6fa8dc'}
MAT_CMAP = 'Greys'        # transition matrices

def use_style():
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'font.size': FS_LABEL, 'axes.linewidth': LW_SPINE,
                         'savefig.bbox': 'tight'})

def row_icon(fig, kind, x, y, size=.034, color=INK):
    """A small human or network glyph, to mark which rows are data and which are model.

    `x`, `y` are the icon's centre in figure coordinates and `size` its width as a figure
    fraction; the height is corrected for the figure's aspect so the glyph stays square.
    Used wherever a figure puts people and networks side by side.
    """
    from matplotlib.patches import Circle, Wedge
    fw, fh = fig.get_size_inches()
    w = size; h = size * fw / fh
    ax = fig.add_axes([x - w / 2, y - h / 2, w, h])
    ax.set(xlim=(-.09, 1.09), ylim=(-.09, 1.09)); ax.axis('off')
    ax.set_aspect('equal')
    if kind == 'human':
        ax.add_patch(Circle((.5, .76), .215, facecolor=color, lw=0, clip_on=False))
        ax.add_patch(Wedge((.5, .26), .42, 0, 180, facecolor=color, lw=0,
                           clip_on=False))
    else:
        nodes = [(.11, .82), (.11, .50), (.11, .18), (.50, .66), (.50, .34), (.89, .50)]
        for a, b in ((0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5)):
            ax.plot(*zip(nodes[a], nodes[b]), color=color, lw=.55, zorder=1,
                    clip_on=False)
        for n in nodes:
            ax.add_patch(Circle(n, .100, facecolor='white', edgecolor=color, lw=.85,
                                zorder=2, clip_on=False))
    return ax


def finish(ax, title=None, legend_fs=None):
    """The plotting contract. Unlike plot_utils.format_axis this leaves line widths alone."""
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ax.spines.values(): sp.set_linewidth(LW_SPINE)
    ax.tick_params(labelsize=FS_TICK, length=LEN_TICK, width=LW_SPINE, pad=1.5)
    ax.xaxis.label.set_fontsize(FS_LABEL); ax.yaxis.label.set_fontsize(FS_LABEL)
    if ax.get_legend() is not None:
        for t in ax.get_legend().get_texts(): t.set_fontsize(legend_fs or FS_TICK)
    if title is not None: ax.set_title(title, fontsize=FS_TITLE, pad=3)

def panel_letter(fig, ax, s, dx=-0.038, dy=0.016):
    fig.text(ax.get_position().x0 + dx, ax.get_position().y1 + dy, s,
             fontsize=FS_TITLE + 1, fontweight='bold', va='bottom', ha='left')

# ----------------------------------------------------------------- data -----
# Where the human CSVs live. Override with the SIMPLICITY_DATA environment variable so the
# code runs unchanged wherever the data sits (e.g. `data/` once this is packaged).
HUMAN_DIR = os.environ.get('SIMPLICITY_DATA') or (
    'data' if os.path.isdir('data') and os.path.exists('data/exp1_test.csv')
    else '../../../data/human_data_final/data')
HUM_FILES = {'standard': 'exp1_learning_control.csv',
             'curriculum': 'exp1_learning_curriculum.csv',
             'anti_curriculum': 'exp1_learning_anticurriculum.csv'}

def load_exp1(human_dir=HUMAN_DIR):
    """-> (LRN dict of learning frames, T test frame). blockNum is made 1-based."""
    LRN = {c: pd.read_csv(f'{human_dir}/{f}', dtype={'stimuli': str})
           for c, f in HUM_FILES.items()}
    T = pd.read_csv(f'{human_dir}/exp1_test.csv', dtype={'stimuli': str})
    for c, d in LRN.items():
        d['block'] = d.blockNum.astype(int) + 1
        d['cond'] = c
    T['cond'] = T.curriculum.map({v: k for k, v in HUMAN_COND.items()})
    assert T.cond.notna().all(), 'test-phase condition labels did not map'
    return LRN, T

# ------------------------------------------------------------ task coding ---
STIM = ['000', '001', '010', '011', '100', '101', '110', '111']
BITS = np.array([[int(s[i]) for i in (0, 1, 2)] for s in STIM])
_rule = BITS[:, 1]                                   # rule = middle feature
_pocket = (BITS[:, 0] == 1) & (BITS[:, 2] == 0)      # the two exceptions
IS_EXC = _pocket
POLICIES = {'rule': _rule, 'rule + exceptions': np.where(_pocket, 1 - _rule, _rule),
            'rule-': 1 - _rule, 'feat0+': BITS[:, 0], 'feat0-': 1 - BITS[:, 0],
            'feat2+': BITS[:, 2], 'feat2-': 1 - BITS[:, 2],
            'const1': np.ones(8, int), 'const0': np.zeros(8, int)}
STATE_OF = {'rule': 'rule', 'rule + exceptions': 'rule + exceptions',
            'rule-': 'other 1-D rule', 'feat0+': 'other 1-D rule',
            'feat0-': 'other 1-D rule', 'feat2+': 'other 1-D rule',
            'feat2-': 'other 1-D rule', 'const1': 'constant', 'const0': 'constant'}
SIDX = {s: i for i, s in enumerate(STIM)}

def classify(k, n):
    """k, n: (8,) responses==1 and trial counts per stimulus -> (state, fitted lapse).

    Deterministic policy + ML lapse, compared with an unpenalised random null by BIC.
    NB the random case must be returned explicitly; falling through a dict lookup
    silently relabels it (this was a real bug in earlier scratch code).
    """
    N = int(n.sum())
    if N == 0: return None, np.nan
    best = ('random', -2 * N * np.log(0.5), np.nan)
    for name, t in POLICIES.items():
        agree = (k * t + (n - k) * (1 - t)).sum(); dis = N - agree
        eps = min(max(dis / N, 1e-6), 0.5)
        bic = -2 * (agree * np.log(1 - eps) + dis * np.log(eps)) + np.log(N)
        if bic < best[1]: best = (name, bic, eps)
    return ('random' if best[0] == 'random' else STATE_OF[best[0]]), best[2]

def _kn(g):
    n = g.groupby('stimuli').size().reindex(STIM, fill_value=0).values
    k = g.groupby('stimuli')['responseCoded'].sum().reindex(STIM, fill_value=0).values
    return k.astype(float), n.astype(float)

def strategy_states_human(LRN, T, conds=CONDS):
    """-> tidy frame (cond, sid, block, state, eps, identifiable). Test gets block = TEST_X."""
    rows = []
    for c in conds:
        for (sid, blk), g in LRN[c].groupby(['subjectId', 'block']):
            k, n = _kn(g); seen = n > 0
            ident = bool((POLICIES['rule'][seen] != POLICIES['rule + exceptions'][seen]).any())
            st, eps = classify(k, n)
            rows.append((c, sid, int(blk), st, eps, ident))
    TEST_X = int(max(r[2] for r in rows)) + 1
    for c in conds:
        for sid, g in T[T.cond == c].groupby('subjectId'):
            k, n = _kn(g); st, eps = classify(k, n)
            rows.append((c, sid, TEST_X, st, eps, True))
    out = pd.DataFrame(rows, columns=['cond', 'sid', 'block', 'state', 'eps', 'identifiable'])
    out.attrs['TEST_X'] = TEST_X
    return out

def state_fractions(STRAT, conds=CONDS):
    """-> {cond: DataFrame(block x state) of percentages, identifiable blocks only}."""
    out = {}
    for c in conds:
        t = (STRAT[(STRAT.cond == c) & STRAT.identifiable]
             .groupby(['block', 'state']).size().unstack(fill_value=0)
             .reindex(columns=STATES, fill_value=0))
        out[c] = 100 * t.div(t.sum(1), axis=0)
    return out

# ------------------------------------------------------- learning curves ----
def learning_curves_human(LRN, bin_size=24, nbin=10, conds=CONDS):
    """Per-participant bin means, then mean/SEM across participants."""
    rows = []
    for c in conds:
        d = LRN[c].copy()
        d['bin'] = np.clip(d.groupby('subjectId').cumcount() // bin_size, 0, nbin - 1) + 1
        pp = d.groupby(['bin', 'stimType', 'subjectId']).correct.mean()
        g = pp.groupby(level=['bin', 'stimType']).agg(['mean', 'sem', 'size'])
        for (b, st), r in g.iterrows():
            rows.append((c, b, st, r['mean'], r['sem'], r['size']))
    return pd.DataFrame(rows, columns=['cond', 'bin', 'stimType', 'mean', 'sem', 'n'])

def test_accuracy_human(T, conds=CONDS):
    pp = T.groupby(['cond', 'stimType', 'subjectId']).correct.mean().reset_index()
    g = pp.groupby(['cond', 'stimType']).correct.agg(['mean', 'sem', 'size']).reset_index()
    return g[g.cond.isin(conds)].rename(columns={'size': 'n'}), pp

# ------------------------------------------------- yoked network (Exp 1) ----
# One network per participant, trained on that participant's exact trial sequence.
# The Experiment 1 model, used by every figure so Fig 2 and Fig 5 are the same nets.
YOKED_EXP1 = dict(hidden=32, lr=0.30, init=0.010, hnoise=1.0, lapse=0.30)

class YokedData:
    """Human trial sequences as tensors, plus the bookkeeping the figures need."""
    def __init__(self, LRN, device='cuda:2', conds=CONDS):
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        rows = []
        for c in conds:
            d = LRN[c].sort_values(['subjectId', 'total_trial_num'])
            for sid, g in d.groupby('subjectId'):
                rows.append((c, sid, g.stimuli.values, g.correctResponse.values))
        self.cond = np.array([r[0] for r in rows]); self.sid = np.array([r[1] for r in rows])
        self.P = len(rows); self.T = max(len(r[2]) for r in rows)
        idx = np.zeros((self.P, self.T), np.int64); lab = np.zeros((self.P, self.T), np.float32)
        msk = np.zeros((self.P, self.T), np.float32)
        for i, (_, _, ss, yy) in enumerate(rows):
            n = len(ss); idx[i, :n] = [SIDX[s] for s in ss]; lab[i, :n] = yy; msk[i, :n] = 1
        self.idx, self.msk = idx, msk.astype(bool)
        self.exc_trial = IS_EXC[idx]
        proto = np.array([[b * 2 - 1 for b in BITS[i]] for i in range(8)], np.float32)
        lab8 = np.zeros(8, np.float32)
        for _, _, ss, yy in rows:
            for s, y in zip(ss, yy): lab8[SIDX[s]] = y
        t = lambda a, d=torch.float32: torch.tensor(a, device=self.dev, dtype=d)
        self.PRt, self.LB8 = t(proto), t(lab8)
        self.IDX, self.LAB, self.MSK = t(idx, torch.long), t(lab), t(msk)
        self.labels8 = lab8

def run_yoked(D, cfg=None, seed=0, block_len=48, ndraw=400, pr_every=4, keep_hidden=(),
              orth=True):
    """Train one net per participant. Returns raw (pre-lapse) quantities.

    pc            (P, T)      P(correct) on the presented stimulus, during training
    proto_blocks  (5, P, 8)   prototype accuracy at each block end, hidden noise
                              marginalised so it is measured like the learning curves
    pr            (n_ck, P)   participation ratio of the 8 hidden representations
    """
    cfg = {**YOKED_EXP1, **(cfg or {})}
    H, LR, INIT, HN = cfg['hidden'], cfg['lr'], cfg['init'], cfg['hnoise']
    g = torch.Generator(device=D.dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, device=D.dev)
    W = (rn(D.P, H, 3) * INIT).requires_grad_(True); b = (rn(D.P, H) * INIT).requires_grad_(True)
    v = (rn(D.P, H) * INIT).requires_grad_(True); par = [W, b, v]
    pc = np.zeros((D.P, D.T)); blocks, prs, pr_t = [], [], []
    seps_r, seps_e = [], []                      # task-axis separations over training
    hid = {}                                     # hidden reps at requested trials, (P, H, 8)
    def snap():
        with torch.no_grad():
            return torch.relu(torch.einsum('nd,phd->phn', D.PRt, W)
                              + b.unsqueeze(-1)).cpu().numpy()
    def probe():
        with torch.no_grad():
            pre = torch.einsum('nd,phd->phn', D.PRt, W) + b.unsqueeze(-1)
            reps = torch.relu(pre).permute(0, 2, 1)
            reps = reps - reps.mean(1, keepdim=True)
            lam = torch.linalg.svdvals(reps) ** 2
            pr = ((lam.sum(-1) ** 2) / lam.pow(2).sum(-1).clamp_min(1e-12)).cpu().numpy()
            tot = torch.zeros(D.P, 8, device=D.dev)
            for _ in range(ndraw):
                h = torch.relu(pre) + rn(D.P, H, 1) * HN
                q = torch.sigmoid(torch.einsum('phn,ph->pn', h, v))
                tot += D.LB8 * q + (1 - D.LB8) * (1 - q)
            # separations are representational, so measured on the noise-free code
            sr, se = axis_separation(torch.relu(pre).cpu().numpy(), orth=orth)
            return pr, (tot / ndraw).cpu().numpy(), sr, se
    for t in range(D.T):
        if t in keep_hidden: hid[t] = snap()
        if t % pr_every == 0:
            pr, _, sr, se = probe()
            prs.append(pr); pr_t.append(t); seps_r.append(sr); seps_e.append(se)
        h = torch.relu(torch.einsum('pd,phd->ph', D.PRt[D.IDX[:, t]], W) + b) + rn(D.P, H) * HN
        lg = (h * v).sum(-1); y = D.LAB[:, t]
        with torch.no_grad():
            p_ = torch.sigmoid(lg); pc[:, t] = (y * p_ + (1 - y) * (1 - p_)).cpu().numpy()
        loss = (F.binary_cross_entropy_with_logits(lg, y, reduction='none') * D.MSK[:, t]).sum()
        gs = torch.autograd.grad(loss, par)
        with torch.no_grad():
            for p2, gr in zip(par, gs): p2 -= LR * gr
        if (t + 1) % block_len == 0: blocks.append(probe()[1])
    pr, acc, sr, se = probe(); prs.append(pr); pr_t.append(D.T)
    seps_r.append(sr); seps_e.append(se)
    if D.T in keep_hidden: hid[D.T] = snap()
    if len(blocks) < 5: blocks.append(acc)
    return dict(pc=pc, proto_blocks=np.stack(blocks[:5]), pr=np.stack(prs),
                pr_t=np.array(pr_t), final=acc, cfg=cfg, hidden=hid,
                sep_rule=np.stack(seps_r), sep_exc=np.stack(seps_e))

def learning_curves_network(D, R, bin_size=24, nbin=10, lapse=None):
    lapse = YOKED_EXP1['lapse'] if lapse is None else lapse
    pcl = (1 - lapse) * R['pc'] + lapse * .5
    bins = np.clip(np.arange(D.T) // bin_size, 0, nbin - 1)
    rows = []
    for c in CONDS:
        ids = np.where(D.cond == c)[0]
        for bnum in range(nbin):
            for st, want in (('Rule', False), ('Exception', True)):
                sel = (bins == bnum)[None, :] & D.msk[ids] & (D.exc_trial[ids] == want)
                n = sel.sum(1); h = n > 0
                if not h.any(): continue
                v = (pcl[ids] * sel).sum(1)[h] / n[h]
                rows.append((c, bnum + 1, st, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)))
    return pd.DataFrame(rows, columns=['cond', 'bin', 'stimType', 'mean', 'sem', 'n'])

def test_accuracy_network(D, R, lapse=None):
    lapse = YOKED_EXP1['lapse'] if lapse is None else lapse
    a = (1 - lapse) * R['final'] + lapse * .5
    rows, per = [], []
    for c in CONDS:
        ids = np.where(D.cond == c)[0]
        for st, m in (('Rule', ~IS_EXC), ('Exception', IS_EXC)):
            v = np.array([a[i][m].mean() for i in ids])
            rows.append((c, st, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)))
            per += [(c, st, D.sid[i], x) for i, x in zip(ids, v)]
    return (pd.DataFrame(rows, columns=['cond', 'stimType', 'mean', 'sem', 'n']),
            pd.DataFrame(per, columns=['cond', 'stimType', 'sid', 'acc']))

def strategy_states_network(D, R, lapse=None, seed=1, reps_block=6, reps_test=2):
    """Sample responses through the lapse at the human trial counts, then the same classifier."""
    lapse = YOKED_EXP1['lapse'] if lapse is None else lapse
    rng = np.random.default_rng(seed); rows = []
    for bi, reps in [(0, reps_block), (1, reps_block), (2, reps_block), (3, reps_block),
                     (4, reps_block), (4, reps_test)]:
        blk = bi + 1 if reps == reps_block else 6
        a = (1 - lapse) * R['proto_blocks'][bi] + lapse * .5
        for i in range(D.P):
            draw = (rng.random((reps, 8)) < a[i]).astype(float)          # 1 = correct
            resp = np.where(D.labels8[None, :] == 1, draw, 1 - draw)
            k = resp.sum(0); n = np.full(8, float(reps))
            st, eps = classify(k, n)
            rows.append((D.cond[i], D.sid[i], blk, st, eps, True))
    return pd.DataFrame(rows, columns=['cond', 'sid', 'block', 'state', 'eps', 'identifiable'])

# --------------------------------------------------- strategy transitions ---
# One transition per consecutive *identifiable* pair of blocks, so the excluded
# block is bridged rather than breaking the chain. The test phase is kept out.
def transition_pairs(STRAT, conds=CONDS, max_block=5):
    SL = (STRAT[STRAT.identifiable & (STRAT.block <= max_block)]
          .sort_values(['cond', 'sid', 'block']))
    piv = {c: SL[SL.cond == c].pivot(index='sid', columns='block', values='state')
           for c in conds}
    XS = {c: list(piv[c].columns) for c in conds}
    TR = pd.concat([pd.DataFrame({'cond': c, 'sid': p.index, 'b0': b0, 'b1': b1,
                                  's0': p[b0].values, 's1': p[b1].values})
                    for c, p in piv.items() for b0, b1 in zip(XS[c], XS[c][1:])],
                   ignore_index=True)
    return TR, piv, XS

def trans_matrix(tr):
    return pd.crosstab(tr.s0, tr.s1).reindex(index=STATES, columns=STATES, fill_value=0)

def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def entry_rates(tr):
    """P(rule + exceptions at the next block | current state), with Wilson intervals."""
    return {st: (int(((tr.s0 == st) & (tr.s1 == 'rule + exceptions')).sum()),
                 int((tr.s0 == st).sum()))
                + wilson(int(((tr.s0 == st) & (tr.s1 == 'rule + exceptions')).sum()),
                         int((tr.s0 == st).sum()))
            for st in STATES}

def _mh(t):
    n = t.sum((-2, -1))
    with np.errstate(invalid='ignore', divide='ignore'):
        w = np.where(n > 0, 1 / np.maximum(n, 1e-9), 0.0)
        return ((t[..., 0, 0] * t[..., 1, 1] * w).sum(-1)
                / (t[..., 0, 1] * t[..., 1, 0] * w).sum(-1))

def stratified_or(tr, by=('b0',), reps=4000, seed=0):
    """Mantel-Haenszel OR for entering rule+exc from `rule` vs the three pre-rule
    states, stratified by `by`, plus a participant-clustered bootstrap CI."""
    import statsmodels.api as sm
    d = tr[tr.s0 != 'rule + exceptions'].copy()
    d['stratum'] = d.groupby(list(by), sort=True).ngroup()
    tabs = np.stack([pd.crosstab(g.s0 == 'rule', g.s1 == 'rule + exceptions')
                     .reindex(index=[True, False], columns=[True, False], fill_value=0).values
                     for _, g in d.groupby('stratum')])
    ct = sm.stats.StratifiedTable(np.dstack(tabs))
    codes, _ = pd.factorize(d.cond + '_' + d.sid.astype(str))
    S = tabs.shape[0]
    A = np.zeros((codes.max() + 1, S, 2, 2))
    np.add.at(A, (codes, d.stratum.values, 1 - (d.s0 == 'rule').values.astype(int),
                  1 - (d.s1 == 'rule + exceptions').values.astype(int)), 1)
    Pn = A.shape[0]
    cnt = np.random.default_rng(seed).multinomial(Pn, np.full(Pn, 1 / Pn), size=reps)
    boot = _mh((cnt @ A.reshape(Pn, -1)).reshape(reps, S, 2, 2))
    return ct, tuple(np.nanpercentile(boot, [2.5, 97.5]))

def transition_summary(STRAT, cond=None, conds=CONDS):
    """Everything Figure 3 needs, for one condition or pooled."""
    TR, piv, XS = transition_pairs(STRAT, conds)
    tr = TR if cond is None else TR[TR.cond == cond]
    by = ('cond', 'b0') if cond is None else ('b0',)
    ct, bci = stratified_or(tr, by)
    return dict(tr=tr, piv=piv, XS=XS, M=trans_matrix(tr), entry=entry_rates(tr),
                OR=ct.oddsratio_pooled, OR_ci=ct.oddsratio_pooled_confint(),
                OR_boot=bci, p=ct.test_null_odds().pvalue, n_tr=len(tr),
                n_sub=tr.sid.nunique())

# ------------------------------------------------------- flow primitive -----
_LANE = {s: i for i, s in enumerate(STATES)}
def _ribbon(ax, x0, x1, y0, y1, h, color, alpha=.5):
    t = np.linspace(0, 1, 60); s = (1 - np.cos(np.pi * t)) / 2
    top = y0 + (y1 - y0) * s
    ax.fill_between(x0 + (x1 - x0) * t, top - h, top, color=color, alpha=alpha, lw=0, zorder=2)

def draw_flow(ax, p, xs, node_w=.13, lane_h=.86):
    """p: participants x block table of states; xs: x position of each column."""
    from matplotlib.patches import Rectangle
    n = len(p); hgt = lambda k: lane_h * k / n
    for x0, x1 in zip(xs, xs[1:]):
        cnt = pd.crosstab(p[x0], p[x1]).reindex(index=STATES, columns=STATES, fill_value=0)
        out = {s: _LANE[s] + hgt((p[x0] == s).sum()) / 2 for s in STATES}
        inn = {s: _LANE[s] + hgt((p[x1] == s).sum()) / 2 for s in STATES}
        for a in STATES[::-1]:
            for b in STATES[::-1]:
                k = cnt.loc[a, b]
                if not k: continue
                h = hgt(k)
                _ribbon(ax, x0 + node_w / 2, x1 - node_w / 2, out[a], inn[b], h, stateColors[a])
                out[a] -= h; inn[b] -= h
    for x in xs:
        for s in STATES:
            k = (p[x] == s).sum()
            if not k: continue
            ax.add_patch(Rectangle((x - node_w / 2, _LANE[s] - hgt(k) / 2), node_w, hgt(k),
                                   fc=stateColors[s], ec='k', lw=.4, zorder=3))

# ----------------------------------- discovery: finding vs keeping the rule --
ON_CUT = 10 / 12 - 1e-9          # "found the rule": <= 2 errors on the 12 rule test trials

def rule_test_accuracy(T, conds=CONDS):
    rt = (T[T.stimType == 'Rule'].groupby(['cond', 'subjectId']).correct
          .agg(['mean', 'size']).reset_index())
    return {c: rt.loc[rt.cond == c, 'mean'].values for c in conds}

def decompose(a_ref, a_new, cut=ON_CUT):
    """Split mean(a_new) - mean(a_ref) into composition (a larger on-rule fraction,
    valued at the reference group's means) and within-learner parts. They sum to the total."""
    on0, on1 = a_ref >= cut, a_new >= cut
    f0, f1 = on0.mean(), on1.mean()
    m_on0, m_off0 = a_ref[on0].mean(), a_ref[~on0].mean()
    m_on1, m_off1 = a_new[on1].mean(), a_new[~on1].mean()
    total = a_new.mean() - a_ref.mean()
    comp = (f1 - f0) * (m_on0 - m_off0)
    within = f1 * (m_on1 - m_on0) + (1 - f1) * (m_off1 - m_off0)
    return total, comp, within, dict(f0=f0, f1=f1, on0=m_on0, on1=m_on1,
                                     off0=m_off0, off1=m_off1)

def decompose_boot(a_ref, a_new, cut=ON_CUT, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([decompose(rng.choice(a_ref, len(a_ref)),
                               rng.choice(a_new, len(a_new)), cut)[:3] for _ in range(B)])

def on_rule_by_block(LRN, T, cut=0.8, conds=CONDS):
    """Per participant x block rule accuracy, plus who is on/off the rule at test."""
    out = {}
    for c in conds:
        d = LRN[c]
        blk = (d[d.stimType == 'Rule'].groupby(['subjectId', 'block']).correct.mean()
               .unstack('block'))
        acc = T[(T.cond == c) & (T.stimType == 'Rule')].groupby('subjectId').correct.mean()
        found = set(acc.index[acc >= ON_CUT])
        out[c] = dict(blk=blk, found=found, frac=(blk >= cut).mean(0))
    return out

# ------------------------------------------------- reaction-time signature ---
RT_LO, RT_HI = 200, 10000

def rt_cost(LRN, conds=CONDS):
    """Per participant x block: mean log-RT, exception minus rule."""
    RT = pd.concat([LRN[c] for c in conds], ignore_index=True)
    RT = RT[(RT.rt >= RT_LO) & (RT.rt <= RT_HI)].copy()
    RT['lrt'] = np.log(RT.rt)
    cost = RT.groupby(['cond', 'subjectId', 'block', 'stimType'])['lrt'].mean().unstack()
    return (cost['Exception'] - cost['Rule']).rename('cost').reset_index().dropna(), RT

def rt_by_state(STRAT, cost, max_block=5):
    """Join the RT cost onto the inferred state, and fit the mixed model."""
    import statsmodels.formula.api as smf, patsy
    bs = (STRAT[STRAT.identifiable & (STRAT.block <= max_block)]
          .merge(cost.rename(columns={'subjectId': 'sid'}), on=['cond', 'sid', 'block'],
                 how='inner'))
    bs['state'] = pd.Categorical(bs.state, categories=STATES)
    bs['uid'] = bs.cond + '_' + bs.sid.astype(str)
    form = 'cost ~ C(state) + block + C(cond)'
    mm = smf.mixedlm(form, bs, groups=bs.uid).fit(reml=True)
    di = patsy.dmatrix(form.split('~')[1].strip(), bs, return_type='dataframe').design_info
    k = len(mm.fe_params); V = np.asarray(mm.cov_params())[:k, :k]
    marg = {}
    for st in STATES:
        f = bs.copy(); f['state'] = pd.Categorical([st] * len(f), categories=STATES)
        cc = np.asarray(patsy.dmatrix(di, f, return_type='dataframe')).mean(0)
        marg[st] = (float(cc @ mm.fe_params), float(np.sqrt(cc @ V @ cc)))
    return bs, mm, marg

def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


# ------------------------------------------------- task-axis projections -----
def task_axes(h, orth=True):
    """h: (P, H, 8) hidden reps -> per-network projections of the 8 stimuli on
    (rule axis, exception axis). Rule axis = class-1 minus class-0 mean over the six
    rule stimuli; exception axis = the direction separating the two exceptions,
    orthogonalised against the rule axis. Both have a fixed sign in every network,
    so projections can be averaged across networks.

    With `orth=False` the exception axis is left as the raw difference between the two
    exceptions, only normalised. The two axes are then oblique, so a projection onto the
    exception axis also contains whatever that direction shares with the rule axis, and
    the separations are no longer independent of one another.
    """
    lab = np.array([POLICIES['rule + exceptions'][i] for i in range(8)])
    rule_ids = np.where(~IS_EXC)[0]; exc_ids = np.where(IS_EXC)[0]
    r1 = [i for i in rule_ids if lab[i] == 1]; r0 = [i for i in rule_ids if lab[i] == 0]
    a1 = h[:, :, r1].mean(-1) - h[:, :, r0].mean(-1)                      # (P, H)
    e_hi = exc_ids[np.argmax(lab[exc_ids])]; e_lo = exc_ids[np.argmin(lab[exc_ids])]
    a2 = h[:, :, e_hi] - h[:, :, e_lo]
    a1n = a1 / np.linalg.norm(a1, axis=1, keepdims=True).clip(1e-9)
    if orth:
        a2 = a2 - (a2 * a1n).sum(1, keepdims=True) * a1n                  # orthogonalise
    a2n = a2 / np.linalg.norm(a2, axis=1, keepdims=True).clip(1e-9)
    return np.einsum('phn,ph->pn', h, a1n), np.einsum('phn,ph->pn', h, a2n)


def axis_separation(h, orth=True):
    """How far apart the classes sit on each task axis, per network.

    sep_rule : class-1 minus class-0 rule stimuli, along the rule axis
    sep_exc  : the class-1 exception minus the class-0 exception, along the
               exception axis (the direction orthogonal to the rule axis)
    """
    x, y = task_axes(h, orth=orth)
    lab = POLICIES['rule + exceptions']
    rid = np.where(~IS_EXC)[0]
    r1 = [i for i in rid if lab[i] == 1]; r0 = [i for i in rid if lab[i] == 0]
    eid = np.where(IS_EXC)[0]
    hi, lo = eid[np.argmax(lab[eid])], eid[np.argmin(lab[eid])]
    return x[:, r1].mean(1) - x[:, r0].mean(1), y[:, hi] - y[:, lo]


def ols_band(x, y, n=100):
    """Least-squares fit with a 95% confidence band, for a ggplot-style overlay."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), n)
    res = y - (a + b * x); dof = len(x) - 2
    s2 = (res ** 2).sum() / dof
    sxx = ((x - x.mean()) ** 2).sum()
    se = np.sqrt(s2 * (1 / len(x) + (xs - x.mean()) ** 2 / sxx))
    from scipy import stats as _s
    t = _s.t.ppf(.975, dof)
    return xs, a + b * xs, (a + b * xs) - t * se, (a + b * xs) + t * se

# ============================== Experiment 2 ================================
E2_CELLS = ['Simple/No Curriculum', 'XOR/No Curriculum',
            'Simple/Simple To Complex', 'XOR/Simple To Complex']
E2_LABEL = {'Simple': '1-D', 'XOR': 'XOR'}
E2_CUR = ['No Curriculum', 'Simple To Complex']

def load_exp2(human_dir=HUMAN_DIR, phase='learning'):
    """The R+E phase of Experiment 2, from the raw file (the tidy exp2_* files drop
    `stimuli` and `responseCoded`, which the strategy classifier needs)."""
    d = pd.read_csv(f'{human_dir}/experiment2_raw.csv', low_memory=False,
                    dtype={'stimuli': str})
    d = d[(d.taskPart == phase) & (~d.timeout.astype(bool))].copy()
    d['block'] = d.blockNum.astype(int) + 1
    d['cur'] = d.curriculum.str.split().str.join(' ')
    d['pre'] = d.pretraining
    d['cell'] = d.pre + '/' + d.cur
    return d

def strategy_states_exp2(d, group='cell'):
    """Same classifier as Experiment 1, applied per participant x block.
    -> tidy frame (cond, sid, block, state, eps, identifiable) with cond = the group key."""
    rows = []
    for (grp, sid, blk), g in d.groupby([group, 'subjectId', 'block']):
        k, n = _kn(g); seen = n > 0
        ident = bool((POLICIES['rule'][seen] != POLICIES['rule + exceptions'][seen]).any())
        st, eps = classify(k, n)
        rows.append((grp, sid, int(blk), st, eps, ident))
    return pd.DataFrame(rows, columns=['cond', 'sid', 'block', 'state', 'eps', 'identifiable'])

def smrt(human_dir=HUMAN_DIR, pretraining='Simple', nbin=5, rule_feature=0):
    """Switch-match reaction time during pretraining, as the Experiment 2 analysis defines it.

    For consecutive trial pairs each of the three features either switches or matches; SMRT
    for a feature is mean RT on switch trials over mean RT on match trials, so SMRT > 1 means
    the participant slowed when that feature changed. `SA` additionally divides a feature's
    SMRT by the mean across the three, which removes the overall speed-up and leaves only
    *relative* attention -- that is the quantity the test-side models use, and the one stored
    as `selective_attention` in exp2_test_smrt.csv.

    Pairs are formed within (subject, block) on consecutive presented trial numbers, so a
    dropped trial breaks the pair instead of creating a false adjacency, and bins are equal
    spans of presented trials. `rule_feature` is the canonical bit the 1-D rule uses (0 for
    every Simple participant). Reproduces the stored per-participant values at r = .996; the
    arithmetic mean RT, the within-block pairing and the absence of an RT range filter are
    all load-bearing, so do not "improve" them without rechecking that number.
    """
    d = pd.read_csv(f'{human_dir}/experiment2_raw.csv', low_memory=False, dtype={'stimuli': str})
    d = d[(d.taskPart == 'metalearning') & (d.pretraining == pretraining)
          & (~d.timeout.astype(bool))].sort_values(['subjectId', 'blockNum', 'trialNum']).copy()
    d['prev_stim'] = d.groupby(['subjectId', 'blockNum']).stimuli.shift(1)
    d['prev_tn'] = d.groupby(['subjectId', 'blockNum']).trialNum.shift(1)
    pair = d[d.trialNum - d.prev_tn == 1]
    span = int(np.ceil((d.trialNum.max() + 1) / nbin))           # presented trials per bin
    long = []
    for f in (0, 1, 2):
        s = pair[['subjectId', 'trialNum', 'rt']].copy()
        s['feature'] = f
        s['switch'] = (pair.stimuli.str[f] != pair.prev_stim.str[f]).values
        long.append(s)
    L = pd.concat(long, ignore_index=True)
    L['bin'] = (L.trialNum // span).astype(int) + 1
    out = (L.groupby(['sid' if False else 'subjectId', 'bin', 'feature', 'switch']).rt.mean()
           .unstack('switch').dropna().rename(columns={False: 'match', True: 'switch'})
           .reset_index())
    out['SMRT'] = out['switch'] / out['match']
    out['SA'] = out.SMRT / out.groupby(['subjectId', 'bin']).SMRT.transform('mean')
    out['relevant'] = out.feature == rule_feature
    out = out.rename(columns={'subjectId': 'sid'})
    out = (out[['sid', 'bin', 'feature', 'match', 'switch', 'SMRT', 'SA', 'relevant']]
           .sort_values(['sid', 'bin', 'feature']).reset_index(drop=True))
    out.attrs['span'] = span
    return out


# ================================================ Experiment 2 yoked model ====
# One network per participant, trained through BOTH phases on that participant's own
# trial sequence, in their own PHYSICAL feature coding. `stimuli` in the raw file is
# LOGICAL coding shared by both phases; `stimulusRandomization` carries two entries,
# 'red' and 'blue'. There is no column saying which is which, so the mapping was
# resolved against `simple_rule_match`: red = pretraining, blue = the R+E phase (and
# test, which follows it), and `featureAssignment` maps logical -> physical. That
# combination reproduces `simple_rule_match` for 100% of the Simple-pretrained
# participants; the three alternatives manage 59-65%. `flipFeatureValues` is taken to
# index physical position -- the repo has no task source to settle it, but it only
# decides which input dimension carries a sign flip, which random symmetric init makes
# immaterial in aggregate.
# alpha is the weight-scale pullback at the phase boundary. It defaults to 0 -- no pullback
# -- because a multi-seed sweep shows it does not produce the key interaction (flat in alpha,
# well inside the seed spread) and it costs the pretraining effect, which decays as alpha
# rises and reverses sign by 0.3. It does improve the fit to the eight test cells, so the
# supplementary figure sweeps it; the main model leaves it out and fits only the two lapses.
# lr_re scales the learning rate in the R+E phase (None = same as pretraining). At 0.5 it
# improves the fit and roughly doubles the pretraining effect, because the networks are
# otherwise too far up the exception curve for the effect to show; it leaves the key
# interaction untouched, like everything else tried. Off by default -- it is a extra
# fitted number, and the main model deliberately inherits Experiment 1 unchanged.
YOKED_EXP2 = dict(hidden=32, lr=0.30, init=0.010, hnoise=1.0,
                  alpha=0.0, lapse_pre=0.10, lapse_re=0.24, lr_re=None)


def _phys8(entry):
    """The 8 logical stimuli as +-1 physical input vectors, for one phase's coding."""
    r = entry['randomization']
    fa = {int(k): v for k, v in r['featureAssignment'].items()}
    flip = list(r['flipFeatureValues'])
    out = np.zeros((8, 3), np.float32)
    for i in range(8):
        bits = np.zeros(3, np.int64)
        for j in range(3):
            bits[fa[j]] = BITS[i][j]
        for p in range(3):
            if flip[p]:
                bits[p] = 1 - bits[p]
        out[i] = bits * 2 - 1
    return out


class YokedExp2Data:
    """Experiment 2 trial sequences as tensors: pretraining, R+E, and test."""

    def __init__(self, human_dir=HUMAN_DIR, device='cuda:2'):
        import ast
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        d = pd.read_csv(f'{human_dir}/experiment2_raw.csv', low_memory=False,
                        dtype={'stimuli': str})
        d = d[~d.timeout.astype(bool)].sort_values(['subjectId', 'blockNum', 'trialNum'])
        PH = {'metalearning': 'p', 'learning': 'r', 'test': 't'}
        sids = np.sort(d.subjectId.unique())
        self.sid = sids
        self.P = len(sids)
        self.pre = np.empty(self.P, object)
        self.cur = np.empty(self.P, object)
        seq = {k: [] for k in 'prt'}; phys8 = []
        for i, sid in enumerate(sids):
            g = d[d.subjectId == sid]
            ent = {e['color']: e for e in ast.literal_eval(g.stimulusRandomization.iloc[0])}
            X = {'p': _phys8(ent['red']), 'r': _phys8(ent['blue']), 't': _phys8(ent['blue'])}
            phys8.append(X)
            self.pre[i] = g.pretraining.iloc[0]
            self.cur[i] = g.curriculum.iloc[0]
            for part, key in PH.items():
                h = g[g.taskPart == part]
                idx = np.array([SIDX[s] for s in h.stimuli], np.int64)
                seq[key].append((X[key][idx], h.correctResponse.values.astype(np.float32),
                                 (h.stimType.values == 'Exception'),
                                 h.trialNum.values.astype(np.int64), idx))
        t = lambda a, dt=torch.float32: torch.tensor(a, device=self.dev, dtype=dt)
        for key in 'prt':
            T = max(len(a[1]) for a in seq[key])
            x = np.zeros((self.P, T, 3), np.float32)
            y = np.zeros((self.P, T), np.float32)
            m = np.zeros((self.P, T), np.float32)
            e = np.zeros((self.P, T), bool)
            tn = np.full((self.P, T), -1, np.int64)     # PRESENTED trial number
            si = np.full((self.P, T), -1, np.int64)     # LOGICAL stimulus index
            for i, (xi, yi, ei, ti, ii) in enumerate(seq[key]):
                n = len(yi); x[i, :n], y[i, :n], m[i, :n], e[i, :n] = xi, yi, 1, ei
                tn[i, :n] = ti; si[i, :n] = ii
            setattr(self, f'tn_{key}', tn); setattr(self, f'sidx_{key}', si)
            setattr(self, f'X{key}', t(x)); setattr(self, f'Y{key}', t(y))
            setattr(self, f'M{key}', t(m)); setattr(self, f'exc_{key}', e)
            setattr(self, f'msk_{key}', m.astype(bool))
        # PR is measured on the 8 corners of the cube, which any coding permutes among
        # themselves, so it does not depend on which phase's coding is used
        self.PRt = t(np.array([[b * 2 - 1 for b in BITS[i]] for i in range(8)], np.float32))
        # ...but the task AXES do depend on it: they ask which stimulus is which, so they
        # need each participant's own physical vectors, indexed by LOGICAL stimulus
        for key in 'prt':
            setattr(self, f'PH_{key}', t(np.stack([d[key] for d in phys8])))   # (P, 8, 3)


def run_yoked_exp2(D2, cfg=None, seed=0, ndraw=400, probe_every=None):
    """Train through pretraining -> (weight-scale pullback) -> R+E, then probe test.

    Returns raw, pre-lapse quantities. Learning curves come from the training forward
    pass (hidden noise injected, i.e. measured as humans are, trial by trial); the test
    probe marginalises the hidden noise over `ndraw` draws.

    With `probe_every` set, also records the participation ratio of the eight stimuli every
    that many trials through BOTH phases, plus the task-axis separations and the noise-free
    per-stimulus logits at the end of each phase. Those use each participant's own physical
    prototypes, because unlike PR the axes depend on which stimulus is which.
    """
    cfg = {**YOKED_EXP2, **(cfg or {})}
    H, LR, INIT, HN, AL = (cfg['hidden'], cfg['lr'], cfg['init'], cfg['hnoise'], cfg['alpha'])
    LR2 = LR if cfg.get('lr_re') is None else cfg['lr_re'] * LR
    g = torch.Generator(device=D2.dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, device=D2.dev)
    W = (rn(D2.P, H, 3) * INIT).requires_grad_(True)
    b = (rn(D2.P, H) * INIT).requires_grad_(True)
    v = (rn(D2.P, H) * INIT).requires_grad_(True)
    par = [W, b, v]
    n0 = [p.detach().flatten(1).norm(dim=1).clone() for p in par]

    def code(PH):                 # (P, H, 8) hidden code for the eight LOGICAL stimuli
        return torch.relu(torch.einsum('pnd,phd->phn', PH, W) + b.unsqueeze(-1))

    def pr_of(h):
        r = h.permute(0, 2, 1); r = r - r.mean(1, keepdim=True)
        lam = torch.linalg.svdvals(r) ** 2
        return ((lam.sum(-1) ** 2) / lam.pow(2).sum(-1).clamp_min(1e-12)).cpu().numpy()

    traj = {'pr': [], 'phase': [], 't': []}

    def train(X, Y, M, lr=None, PH=None, tag=''):
        lr = LR if lr is None else lr
        pc = np.zeros((D2.P, X.shape[1]))
        for t in range(X.shape[1]):
            if probe_every and t % probe_every == 0:
                with torch.no_grad():
                    traj['pr'].append(pr_of(code(PH))); traj['phase'].append(tag)
                    traj['t'].append(t)
            h = torch.relu(torch.einsum('pd,phd->ph', X[:, t], W) + b) + rn(D2.P, H) * HN
            lg = (h * v).sum(-1); y = Y[:, t]
            with torch.no_grad():
                q = torch.sigmoid(lg); pc[:, t] = (y * q + (1 - y) * (1 - q)).cpu().numpy()
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction='none') * M[:, t]).sum()
            gs = torch.autograd.grad(loss, par)
            with torch.no_grad():
                for p_, gr in zip(par, gs):
                    p_ -= lr * gr
        return pc

    def participation_ratio():
        with torch.no_grad():
            pre = torch.einsum('nd,phd->phn', D2.PRt, W) + b.unsqueeze(-1)
            reps = torch.relu(pre).permute(0, 2, 1)
            reps = reps - reps.mean(1, keepdim=True)
            lam = torch.linalg.svdvals(reps) ** 2
            return ((lam.sum(-1) ** 2) / lam.pow(2).sum(-1).clamp_min(1e-12)).cpu().numpy()

    pc_pre = train(D2.Xp, D2.Yp, D2.Mp, PH=D2.PH_p, tag='pre')
    pr = participation_ratio()
    with torch.no_grad():        # pull the weight scale a fraction AL back toward its init
        for p_, nn in zip(par, n0):
            n = p_.flatten(1).norm(dim=1).clamp_min(1e-12)
            p_ *= ((nn / n) ** AL).reshape((-1,) + (1,) * (p_.dim() - 1))
    if probe_every:
        with torch.no_grad():
            h_ = code(D2.PH_r); extra = dict(pr_boundary=pr_of(h_))
    pc_re = train(D2.Xr, D2.Yr, D2.Mr, LR2, PH=D2.PH_r, tag='re')
    with torch.no_grad():
        pre = torch.einsum('ptd,phd->pht', D2.Xt, W) + b.unsqueeze(-1)
        tot = torch.zeros(D2.P, D2.Xt.shape[1], device=D2.dev)
        for _ in range(ndraw):
            h = torch.relu(pre) + rn(D2.P, H, 1) * HN
            q = torch.sigmoid(torch.einsum('pht,ph->pt', h, v))
            tot += D2.Yt * q + (1 - D2.Yt) * (1 - q)
        pc_test = (tot / ndraw).cpu().numpy()
    out = dict(pc_pre=pc_pre, pc_re=pc_re, pc_test=pc_test, pr=pr, cfg=cfg)
    if probe_every:
        with torch.no_grad():
            h = code(D2.PH_r)
            extra['pr_end'] = pr_of(h)
            extra['logit'] = torch.einsum('phn,ph->pn', h, v).cpu().numpy()
            hn = h.cpu().numpy()
        extra['sep_rule'], extra['sep_exc'] = axis_separation(hn)
        extra['traj'] = np.stack(traj['pr'])
        extra['traj_phase'] = np.array(traj['phase'])
        extra['traj_t'] = np.array(traj['t'])
        out.update(extra)
    return out


def _binned(pc, msk, tn, nbin=10, sel=None):
    """Mean P(correct) per participant per bin, over trials selected by `sel`.

    Bins are equal spans of PRESENTED trial number, not of rank among surviving trials.
    That distinction matters: a participant who lost a trial to a timeout has every later
    stimulus shift one rank earlier, which leaks a handful of block-2 exception trials into
    the rank-based bin 2 and puts a stray point on the simple-to-complex exception curve
    where the design guarantees there are none.
    """
    keep = msk if sel is None else (msk & sel)
    span = int(np.ceil((tn.max() + 1) / nbin))
    idx = np.where(tn >= 0, tn // span, -1)
    out = np.full((pc.shape[0], nbin), np.nan)
    for k in range(nbin):
        w = keep & (idx == k)
        n = w.sum(1)
        out[:, k] = np.where(n > 0, (pc * w).sum(1) / np.maximum(n, 1), np.nan)
    return out


def exp2_curves(D2, R2, nbin=10, lapse_pre=None, lapse_re=None):
    """Tidy network curves for Figure 5, with the per-phase lapse applied.

    Empty bins are NaN and plot as gaps: under simple-to-complex there are genuinely no
    exception trials in the first block, so there is nothing to average.
    """
    import warnings
    warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
    c = R2['cfg']
    lp = c['lapse_pre'] if lapse_pre is None else lapse_pre
    lr_ = c['lapse_re'] if lapse_re is None else lapse_re
    lap = lambda a, e: (1 - e) * a + e / 2
    def _ms(a):
        """Column-wise mean and SEM over units, ignoring NaN (empty bins)."""
        n = np.sum(~np.isnan(a), 0)
        return (np.nanmean(a, 0),
                np.nanstd(a, 0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan)))
    out = {}
    _pb = lap(_binned(R2['pc_pre'], D2.msk_p, D2.tn_p, nbin), lp)
    out['pre'], out['pre_sem'] = {}, {}
    for p in ('Simple', 'XOR'):
        out['pre'][p], out['pre_sem'][p] = _ms(_pb[D2.pre == p])
    for cur in E2_CUR:
        for p in ('Simple', 'XOR'):
            m = (D2.pre == p) & (D2.cur == cur)
            for nm, sel in (('Rule', ~D2.exc_r), ('Exception', D2.exc_r)):
                out[(cur, p, nm)], out[('sem', cur, p, nm)] = _ms(
                    lap(_binned(R2['pc_re'], D2.msk_r, D2.tn_r, nbin, sel), lr_)[m])
            for nm, sel in (('Rule', ~D2.exc_t), ('Exception', D2.exc_t)):
                v = lap(np.where(D2.msk_t & sel, R2['pc_test'], np.nan), lr_)
                per = np.nanmean(v[m], 1)                  # one value per network
                per = per[~np.isnan(per)]
                out[('test', cur, p, nm)] = per.mean()
                out[('test_sem', cur, p, nm)] = per.std(ddof=1) / np.sqrt(len(per))
    out['pr'] = {p: R2['pr'][D2.pre == p].mean() for p in ('Simple', 'XOR')}
    return out


def exp2_human_curves(human_dir=HUMAN_DIR, nbin=10):
    """The matching human curves, measured the same way."""
    d = pd.read_csv(f'{human_dir}/experiment2_raw.csv', low_memory=False,
                    dtype={'stimuli': str})
    d = d[~d.timeout.astype(bool)].sort_values(['subjectId', 'blockNum', 'trialNum'])
    # bin by PRESENTED trial number, for the reason given in `_binned`
    d['span'] = d.groupby('taskPart').trialNum.transform(lambda t: np.ceil((t.max() + 1) / nbin))
    d['bin'] = np.minimum((d.trialNum // d.span).astype(int), nbin - 1) + 1
    out = {}
    m = d[d.taskPart == 'metalearning']
    out['pre'], out['pre_sem'] = {}, {}
    for p in ('Simple', 'XOR'):
        _g = m[m.pretraining == p].groupby(['subjectId', 'bin']).correct.mean().groupby('bin')
        out['pre'][p] = _g.mean().reindex(range(1, nbin + 1)).values
        out['pre_sem'][p] = _g.sem().reindex(range(1, nbin + 1)).values
    L, T = d[d.taskPart == 'learning'], d[d.taskPart == 'test']
    for cur in E2_CUR:
        for p in ('Simple', 'XOR'):
            for nm in ('Rule', 'Exception'):
                q = L[(L.curriculum == cur) & (L.pretraining == p) & (L.stimType == nm)]
                _g = q.groupby(['subjectId', 'bin']).correct.mean().groupby('bin')
                out[(cur, p, nm)] = _g.mean().reindex(range(1, nbin + 1)).values
                out[('sem', cur, p, nm)] = _g.sem().reindex(range(1, nbin + 1)).values
                r = T[(T.curriculum == cur) & (T.pretraining == p) & (T.stimType == nm)]
                s = r.groupby('subjectId').correct.mean()
                out[('test', cur, p, nm)] = s.mean()
                out[('test_sem', cur, p, nm)] = s.sem()
    return out


def exp2_test_accuracy_long(D2, R2, lapse_re=None):
    """Per-network test accuracy, one row per (network, stimulus type).

    This keeps the network's CONTINUOUS P(correct) rather than sampling a binary response.
    Sampling 16 binary trials per network -- to mirror how a participant is measured -- is
    a fair way to ask "would an experiment this size detect the effect", but it discards
    most of what the model knows and costs an order of magnitude in power. The continuous
    form is the one the earlier non-yoked modelling reported, and the one to use when the
    question is whether the effect is present in the model at all.
    """
    eps = R2['cfg']['lapse_re'] if lapse_re is None else lapse_re
    pr = (1 - eps) * R2['pc_test'] + eps / 2
    rows = []
    for name, sel in (('Rule', ~D2.exc_t), ('Exception', D2.exc_t)):
        w = D2.msk_t & sel
        acc = np.where(w.sum(1) > 0, (pr * w).sum(1) / np.maximum(w.sum(1), 1), np.nan)
        rows.append(pd.DataFrame(dict(
            acc=acc, unit=np.arange(D2.P), stimType=name,
            pre01=(D2.pre == 'Simple').astype(int),
            cur01=(D2.cur == 'Simple To Complex').astype(int),
            exc01=int(name == 'Exception'))))
    return pd.concat(rows, ignore_index=True).dropna(subset=['acc'])


def exp2_threeway(df, y='acc', ols=True):
    """pretraining x curriculum x stimulus-type on test accuracy, SEs clustered by unit.

    Returns (three-way, pre x cur on rule, pre x cur on exceptions), each (b, t, p). The
    three-way is how much larger the pretraining x curriculum interaction is on exception
    than on rule stimuli; the third is the sum of the other two, which is the contrast the
    Experiment 2 text leads with.
    """
    import statsmodels.api as sm, statsmodels.formula.api as smf
    fam = dict(family=sm.families.Binomial()) if not ols else {}
    fit = (smf.ols if ols else smf.glm)(f'{y} ~ pre01*cur01*exc01', data=df, **fam).fit(
        cov_type='cluster', cov_kwds={'groups': df.unit})
    k3, k2 = 'pre01:cur01:exc01', 'pre01:cur01'
    L = np.zeros(len(fit.params))
    L[list(fit.params.index).index(k2)] = 1
    L[list(fit.params.index).index(k3)] = 1
    t = fit.t_test(L)
    return ((fit.params[k3], fit.tvalues[k3], fit.pvalues[k3]),
            (fit.params[k2], fit.tvalues[k2], fit.pvalues[k2]),
            (float(t.effect), float(t.tvalue), float(t.pvalue)))


# ------------------------------- strategy-state inferential statistics ------
# Added for the Figure 2 section. Three families, each with its own dependence
# structure: (i) two cells of one multinomial within a block, (ii) independent
# samples across conditions at test, (iii) transitions, several per participant.

def wilson(k, n, z=1.96):
    """Wilson score interval for a single proportion."""
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def prop_contrast(k1, n1, k0, n0):
    """Difference between two INDEPENDENT proportions.

    Newcombe's score ('method 10') interval for the difference, which behaves well with
    small or zero cells, plus Fisher's exact p and an odds ratio (Haldane-Anscombe
    corrected when a cell is empty, so a 0/n cell gives a finite estimate).
    """
    from scipy import stats as _s
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1); l0, u0 = wilson(k0, n0)
    lo = (p1 - p0) - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = (p1 - p0) + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    tab = [[k1, n1 - k1], [k0, n0 - k0]]
    p = _s.fisher_exact(tab)[1]
    a, b, c, d = k1, n1 - k1, k0, n0 - k0
    corr = 0.5 if min(a, b, c, d) == 0 else 0.0
    orr = ((a + corr) * (d + corr)) / ((b + corr) * (c + corr))
    return dict(p1=p1, p0=p0, diff=p1 - p0, lo=lo, hi=hi, p=p, OR=orr,
                haldane=corr > 0, k1=int(k1), n1=int(n1), k0=int(k0), n0=int(n0))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvals, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(run, 1.0)
    return adj


def conditional_binomial(STRAT, cond, block, a='rule', b='rule + exceptions'):
    """Of the participants in state `a` or `b` at `block`, how many are in `a`?

    The five states are cells of ONE multinomial per participant-block, so two of them
    cannot be compared as independent proportions. Conditioning on being in either leaves
    an exact binomial test against 0.5.
    """
    from scipy import stats as _s
    g = STRAT[(STRAT.cond == cond) & (STRAT.block == block) & STRAT.identifiable]
    ka, kb = int((g.state == a).sum()), int((g.state == b).sum())
    n = ka + kb
    lo, hi = wilson(ka, n)
    return dict(k=ka, n=n, frac=ka / n if n else np.nan, lo=lo, hi=hi,
                p=_s.binomtest(ka, n, .5).pvalue, n_block=int(len(g)))


def state_table_test(STRAT, block, states, conds=CONDS, reps=20000, seed=0):
    """Curriculum x {in `states`, not} at one block -> omnibus test across the conditions.

    Returns a dict with the table, chi-square, df, p, Cramer's V and the smallest expected
    count. Uses the asymptotic chi-square when every expected count is at least 5 and a
    Monte-Carlo permutation of the condition labels otherwise; `exact` says which was used.
    No continuity correction: Yates applies only to 2 x 2 tables and is conservative.
    """
    from scipy import stats as _s
    rows, labs = [], []
    for c in conds:
        g = STRAT[(STRAT.cond == c) & (STRAT.block == block) & STRAT.identifiable]
        if len(g) == 0: continue
        k = int(g.state.isin(states).sum()); rows.append([k, len(g) - k]); labs.append(c)
    tab = np.array(rows)
    chi2, p, dof, exp = _s.chi2_contingency(tab, correction=False)
    exact = exp.min() < 5
    if exact:
        lab = np.repeat(np.arange(len(tab)), tab.sum(1))
        hit = np.concatenate([np.r_[np.ones(a_), np.zeros(b_)] for a_, b_ in tab])
        rng = np.random.default_rng(seed); null = np.empty(reps)
        for i in range(reps):
            h = rng.permutation(hit)
            t = np.stack([[h[lab == j].sum(), (1 - h[lab == j]).sum()]
                          for j in range(len(tab))])
            null[i] = _s.chi2_contingency(t, correction=False)[0]
        p = (1 + (null >= chi2).sum()) / (reps + 1)
    n = int(tab.sum())
    return dict(tab=tab, labs=labs, chi2=chi2, dof=int(dof), p=p, n=n,
                V=np.sqrt(chi2 / (n * (min(tab.shape) - 1))), min_exp=exp.min(), exact=exact)


def mh_curriculum_or(TR, s0_state, focus='curriculum', target='rule + exceptions',
                     reps=4000, seed=0):
    """Mantel-Haenszel OR contrasting ONE curriculum against the other two.

    Restricted to transitions leaving `s0_state`, stratified by block, with a
    participant-clustered bootstrap CI. Unlike `stratified_or` -- which compares starting
    states within a condition and is NOT a curriculum contrast -- the rows here are the
    curriculum. Use this for any claim of the form "curriculum X does this more than the
    others".
    """
    import statsmodels.api as sm
    d = TR[TR.s0 == s0_state].copy()
    d['foc'] = d.cond == focus
    d['hit'] = d.s1 == target
    d['stratum'] = d.groupby('b0', sort=True).ngroup()
    tabs = np.stack([pd.crosstab(g.foc, g.hit)
                     .reindex(index=[True, False], columns=[True, False], fill_value=0).values
                     for _, g in d.groupby('stratum')])
    ct = sm.stats.StratifiedTable(np.dstack(tabs))
    codes, _ = pd.factorize(d.cond + '_' + d.sid.astype(str))
    S = tabs.shape[0]
    A = np.zeros((codes.max() + 1, S, 2, 2))
    np.add.at(A, (codes, d.stratum.values, 1 - d.foc.values.astype(int),
                  1 - d.hit.values.astype(int)), 1)
    Pn = A.shape[0]
    cnt = np.random.default_rng(seed).multinomial(Pn, np.full(Pn, 1 / Pn), size=reps)
    boot = _mh((cnt @ A.reshape(Pn, -1)).reshape(reps, S, 2, 2))
    rates = {c: float((TR[(TR.cond == c) & (TR.s0 == s0_state)].s1 == target).mean())
             for c in CONDS}
    n_by = {c: int(((TR.cond == c) & (TR.s0 == s0_state)).sum()) for c in CONDS}
    return dict(OR=ct.oddsratio_pooled, OR_ci=ct.oddsratio_pooled_confint(),
                p=ct.test_null_odds().pvalue,
                OR_boot=tuple(np.nanpercentile(boot, [2.5, 97.5])),
                rates=rates, n=n_by, n_tr=len(d), n_sub=d.sid.nunique())


# --------------------------- test-phase models, both experiments ------------
# One row per (network, stimulus): the noise-marginalised P(correct) for that stimulus,
# with the group lapse applied. This is the finest grain the network has at test -- there
# are no trials, only an expectation per stimulus over the hidden-noise draws -- and it is
# the analogue of the trial-level human data. The model below is the analogue of the human
# glmer(correct ~ ... * stimType + (stimType | subjectId)).

def test_long_exp1(D, R, lapse=None):
    """-> tidy frame (net, sid, cond, stim, is_exc, p) + the model's coded columns."""
    lapse = YOKED_EXP1['lapse'] if lapse is None else lapse
    a = (1 - lapse) * R['final'] + lapse * .5
    df = pd.DataFrame([(i, D.sid[i], D.cond[i], s, bool(IS_EXC[s]), a[i, s])
                       for i in range(D.P) for s in range(8)],
                      columns=['net', 'sid', 'cond', 'stim', 'is_exc', 'p'])
    df['exc_c'] = np.where(df.is_exc, .5, -.5)        # sum coding: rule -.5, exception +.5
    df['sc'] = (df.cond == 'curriculum').astype(int)
    df['cs'] = (df.cond == 'anti_curriculum').astype(int)
    return df


def test_long_exp2(D2, R2, lapse_re=None):
    """As `test_long_exp1`, for Experiment 2. Averages the two presentations per stimulus."""
    eps = R2['cfg']['lapse_re'] if lapse_re is None else lapse_re
    pr = (1 - eps) * R2['pc_test'] + eps / 2
    rows = []
    for i in range(D2.P):
        w = D2.msk_t[i]
        if not w.any(): continue                       # a few networks have no test trials
        si = D2.sidx_t[i]
        for s in range(8):
            m = w & (si == s)
            if m.any():
                rows.append((i, D2.sid[i], D2.pre[i], D2.cur[i], s, bool(IS_EXC[s]),
                             float(pr[i][m].mean())))
    df = pd.DataFrame(rows, columns=['net', 'sid', 'pre', 'cur', 'stim', 'is_exc', 'p'])
    df['exc_c'] = np.where(df.is_exc, .5, -.5)
    df['pre01'] = (df.pre == 'Simple').astype(int)     # 1 = one-dimensional pretraining
    df['cur01'] = (df.cur == E2_CUR[1]).astype(int)    # 1 = simple-to-complex
    return df


def test_lmm(df, formula):
    """Linear mixed model with a per-network intercept AND slope on stimulus type.

    `re_formula='~exc_c'` gives an unstructured 2 x 2 covariance for (intercept, slope),
    which -- unlike a random intercept alone -- can represent the NEGATIVE within-network
    correlation between rule and exception accuracy that the rule/exception trade-off
    produces. A random-intercept-only model is misspecified here: its variance component
    is estimated as exactly zero.
    """
    import statsmodels.formula.api as smf
    return smf.mixedlm(formula, df, groups=df.net, re_formula='~exc_c').fit()


def lmm_table(m, drop_re=True):
    """-> DataFrame of b, 95% CI, z, p for the fixed effects, in model order."""
    ci = m.conf_int()
    k = [i for i in m.params.index if not (drop_re and ('Var' in i or 'Cov' in i))]
    return pd.DataFrame({'b': m.params[k], 'lo': ci.loc[k, 0], 'hi': ci.loc[k, 1],
                         'z': m.tvalues[k], 'p': m.pvalues[k]})


def lmm_contrast(m, weights):
    """A linear contrast of an LMM's fixed effects -> (b, lo, hi, z, p).

    `weights` maps term name to coefficient, e.g. {'sc': 1, 'sc:exc_c': .5} for the simple
    effect of the curriculum on exception stimuli when exc_c is sum-coded at +0.5. This is
    the analogue of the within-stimulus-type contrasts the human models report.
    """
    fe = list(m.params.index[:m.k_fe])
    v = np.zeros((1, m.k_fe))
    for k, c in weights.items():
        v[0, fe.index(k)] = c
    t = m.t_test(v)
    b, se = float(np.squeeze(t.effect)), float(np.squeeze(t.sd))
    return b, b - 1.96 * se, b + 1.96 * se, float(np.squeeze(t.tvalue)), \
        float(np.squeeze(t.pvalue))


# ------------------------------------------------------- result caching -----
# Two ways to run the pipeline:
#   replay    -- figures and statistics from cached network outputs (minutes, CPU only)
#   recompute -- retrain the yoked models, rewrite the caches, then replay
# `cached()` is the switch. Caches are .npz rather than pickle so they stay readable
# without this module, and so a reviewer can inspect them.

CACHE_DIR = 'results/cache'


def _pack(d, out, meta, prefix=''):
    """Flatten a results dict into arrays + a JSON-able remainder."""
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, np.ndarray):
            out[key] = v
        elif isinstance(v, dict) and v and all(isinstance(x, np.ndarray) for x in v.values()):
            meta['arraydict'][key] = [[str(kk), type(kk).__name__] for kk in v]
            for kk, vv in v.items():
                out[f'{key}::{kk}'] = vv
        else:
            meta['plain'][key] = v


def _unpack(z, meta):
    out = {}
    for key, spec in meta['arraydict'].items():
        out[key] = {(int(kk) if t == 'int' else kk): z[f'{key}::{kk}'] for kk, t in spec}
    taken = {f'{k}::{kk}' for k, spec in meta['arraydict'].items() for kk, _ in spec}
    for key in z.files:
        if key != '__meta__' and key not in taken:
            out[key] = z[key]
    out.update(meta['plain'])
    return out


def cache_path(name, cache_dir=CACHE_DIR):
    return os.path.join(cache_dir, f'{name}.npz')


def cached(name, fn, force=False, cache_dir=CACHE_DIR, verbose=True):
    """Return fn()'s result, reading or writing results/cache/<name>.npz.

    `force=True` recomputes and overwrites, which is what the recompute entry point does.
    Set the environment variable SIMPLICITY_RECOMPUTE=1 to force every cache in a run.
    """
    import json as _json
    p = cache_path(name, cache_dir)
    force = force or os.environ.get('SIMPLICITY_RECOMPUTE', '') not in ('', '0')
    if not force and os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            meta = _json.loads(str(z['__meta__'].item()))
            if verbose:
                print(f'  [cache] {name}: loaded {p}')
            return _unpack(z, meta)
    if verbose:
        print(f'  [cache] {name}: computing (this is the slow path)...', flush=True)
    res = fn()
    os.makedirs(cache_dir, exist_ok=True)
    out, meta = {}, {'arraydict': {}, 'plain': {}}
    _pack(res, out, meta)
    out['__meta__'] = np.array(_json.dumps(meta, default=float))
    np.savez_compressed(p, **out)
    if verbose:
        print(f'  [cache] {name}: wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)')
    return res
