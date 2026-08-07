"""Runner: compute the blob-curriculum sweeps and pickle them for the notebook.

Usage
-----
    python run_blob_experiments.py                 # full run, auto GPU
    python run_blob_experiments.py --quick         # tiny smoke test
    python run_blob_experiments.py --device cpu --num-seeds 64 --max-steps 2000
    python run_blob_experiments.py --optimizer adam --lr 0.02 --only main_sweep

Outputs (default ./results/): main_sweep.pkl, hidden_sweep.pkl, metadata.json.
The analysis notebook only reads these.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import blob_experiments as exp
from blob_task import BlobTask

EXPERIMENTS = {
    "main_sweep": exp.main_sweep,
    "hidden_sweep": exp.hidden_sweep,
    "init_scale_sweep": exp.init_scale_sweep,
}


def build_config(args) -> exp.BlobConfig:
    config = exp.BlobConfig()
    if args.num_seeds is not None:
        config.num_seeds = args.num_seeds
    if args.max_steps is not None:
        config.max_steps = args.max_steps
        config.checkpoints = tuple(range(args.max_steps // 20, args.max_steps + 1,
                                         max(1, args.max_steps // 20)))
    if args.n_hidden is not None:
        config.n_hidden = args.n_hidden
    if args.optimizer is not None:
        config.optimizer = args.optimizer
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.use_output_bias is not None:
        config.use_output_bias = args.use_output_bias
    if args.accuracy_mode is not None:
        config.accuracy_mode = args.accuracy_mode
    if args.quick:
        config.num_seeds = args.num_seeds or 32
        config.max_steps = args.max_steps or 800
        config.checkpoints = tuple(range(100, 801, 100))
        config.noise_levels = (0.0, 0.5, 1.0, 1.5, 2.0)
        config.n_hidden_grid = (2, 8, 32)
        config.init_scale_grid = (0.1, 0.5, 2.0)
        config.representative_noise = (0.0, 1.0)
    return config


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-hidden", type=int, default=None,
                        help="hidden width for the main sweep")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=None)
    parser.add_argument("--learning-rate", "--lr", type=float, default=None,
                        dest="learning_rate")
    parser.add_argument("--output-bias", dest="use_output_bias",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="include a learnable readout/output bias "
                             "(--output-bias / --no-output-bias; default: off)")
    parser.add_argument("--accuracy-mode", choices=["argmax", "expected"],
                        default=None,
                        help="'argmax' (hard threshold) or 'expected' (soft, "
                             "sigmoid probability of the true label)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = resolve_device(args.device)
    config = build_config(args)
    task = BlobTask(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  seeds/cell: {config.num_seeds}  "
          f"max_steps: {config.max_steps}  width: {config.n_hidden}  "
          f"noise levels: {list(config.noise_levels)}")

    to_run = args.only or list(EXPERIMENTS)
    timings = {}
    for name in to_run:
        t0 = time.time()
        print(f"\n[{name}] running ...", flush=True)
        result = EXPERIMENTS[name](config, task)
        with open(output_dir / f"{name}.pkl", "wb") as fh:
            pickle.dump(result, fh)
        timings[name] = round(time.time() - t0, 1)
        print(f"[{name}] done in {timings[name]}s -> {name}.pkl")

    metadata = dict(device=str(device), seeds_per_cell=config.num_seeds,
                    config=config.to_json_dict(), experiments_run=to_run,
                    timings_seconds=timings,
                    note="seeds_per_cell = nets averaged per (condition, noise) "
                         "cell. signal_noise_std=0 recovers the 8 fixed examples.")
    with open(output_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\nAll done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
