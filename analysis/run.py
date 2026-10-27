"""Entry point for reproducing the paper's figures and statistics.

    python run.py figures     replay: figures + statistics from the cached network outputs
    python run.py model       recompute: retrain the yoked networks, then replay
    python run.py sweep       the two hyperparameter sweeps (slow; rarely needed)
    python run.py all         sweep, then model, then figures

`figures` is the normal path. It needs only the human CSVs and results/cache/*.npz, runs on
CPU in a few minutes, and reproduces every figure and results/main_text_stats.{md,json}.

`model` deletes results/cache/*.npz and retrains both yoked models -- one network per
participant, on that participant's own trial sequence -- then rewrites the caches and runs
the figure pass. This needs a GPU and takes roughly half an hour. Use it to verify that the
cached outputs really are what the code produces.

Network training is seeded (seed 0 throughout), so `model` reproduces the cached values
exactly on the same hardware. Across GPUs, small floating-point differences are expected;
the reported effects are far larger than that.
"""
import argparse, glob, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))     # analysis/
ROOT = os.path.dirname(HERE)                          # the repository root
NB = os.path.join(HERE, 'paper_figures.ipynb')
CACHE = os.path.join(ROOT, 'results', 'cache')


def run_notebook():
    """Execute every code cell in order, in one namespace -- the same thing opening the
    notebook and running all cells does, without needing Jupyter installed."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    nb = json.load(open(NB))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    g = {'__name__': '__main__'}
    os.chdir(ROOT)            # data/, results/ and figures/ are repo-root relative
    sys.path.insert(0, HERE)  # ... while the library lives in analysis/
    for i, c in enumerate(cells, 1):
        src = ''.join(c['source'])
        print(f'--- cell {i}/{len(cells)} ---', flush=True)
        exec(compile(src, f'<cell {i}>', 'exec'), g)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['figures', 'model', 'sweep', 'controls', 'all'])
    ap.add_argument('--seeds', type=int, default=10, help='seeds for sweep_benefit')
    a = ap.parse_args()
    t0 = time.time()

    if a.mode in ('sweep', 'all'):
        for script, extra in (('reconstruct_sweep.py', []),
                              ('sweep_benefit.py', ['--seeds', str(a.seeds)])):
            print(f'=== {script} ===', flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, script)] + extra,
                           cwd=ROOT, check=True)

    if a.mode in ('controls', 'all'):
        # the two control simulations reported in the supplement, then their figures
        for script, extra in (('consolidation_control.py', ['--seeds', '20']),
                              ('lazy_vs_rich_exp2.py', ['--seeds', '3']),
                              ('make_supp_figures.py', [])):
            print(f'=== {script} ===', flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, script)] + extra,
                           cwd=ROOT, check=True)

    if a.mode in ('model', 'all'):
        stale = glob.glob(os.path.join(CACHE, '*.npz'))
        for p in stale:
            os.remove(p)
        print(f'=== recompute: cleared {len(stale)} cached model outputs ===', flush=True)

    if a.mode in ('figures', 'model', 'all'):
        n = len(glob.glob(os.path.join(CACHE, '*.npz')))
        how = f'replaying {n} cached models' if n else 'no cache, models will be trained'
        print(f'=== figures ({how}) ===', flush=True)
        run_notebook()

    print(f'\ndone in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
