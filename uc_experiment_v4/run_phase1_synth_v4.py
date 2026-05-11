"""
V4 Phase 1: 2^4 factorial × synthetic symmetry-sweep instances.

Grid: N_g ∈ {24, 48}; k_identical ∈ {1, 4, 8, 16, 24}; T = 24; seeds [42..44].
For N_g=48, k_identical caps at min(k, N_g).
All 16 feature combinations run on every instance.

Output: uc_experiment_v4/phase1_v4_results.csv (atomic checkpoint per run).
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from v4_common import _synth_worker, checkpoint_csv, force_single_thread_env  # noqa: E402

OUTPUT_CSV = os.path.join(THIS_DIR, 'phase1_v4_results.csv')
N_G_GRID = [24, 48]
K_GRID = [1, 4, 8, 16, 24]
HORIZON = 24
SEEDS = [42, 43, 44]
TIME_LIMIT = 300.0


def build_tasks(quick=False):
    sizes = [24] if quick else N_G_GRID
    seeds = [42] if quick else SEEDS
    k_grid = [1, 8] if quick else K_GRID
    horizon = 12 if quick else HORIZON
    time_limit = 30.0 if quick else TIME_LIMIT
    combos = [(0, 0, 0, 0), (1, 1, 1, 1)] if quick else list(product([0, 1], repeat=4))

    tasks = []
    for N_g in sizes:
        for k in k_grid:
            if k > N_g:
                continue
            for seed in seeds:
                for combo in combos:
                    feats = dict(zip(['F1', 'F2', 'F3', 'F4'],
                                     [bool(x) for x in combo]))
                    tasks.append({
                        'N_g': N_g, 'k_identical': k,
                        'horizon': horizon, 'seed': seed,
                        'features': feats,
                        'time_limit': time_limit, 'gap_tol': 0.001,
                    })
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', default=OUTPUT_CSV)
    args = parser.parse_args()

    force_single_thread_env()
    tasks = build_tasks(quick=args.quick)
    print(f"V4 Phase 1 (synth): {len(tasks)} runs, "
          f"{args.workers} workers, output={args.output}", flush=True)

    rows = []
    t0 = time.perf_counter()
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            r = _synth_worker(task)
            rows.append(r); checkpoint_csv(rows, args.output)
            print(f"[{i:4d}/{len(tasks)}] {r.get('instance_id'):28s} "
                  f"feats={r.get('feature_label'):4s} "
                  f"nodes={r.get('mip_node_count')} "
                  f"time={r.get('mip_total_time_s')} "
                  f"status={r.get('status')}", flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_synth_worker, tasks), 1):
                rows.append(r); checkpoint_csv(rows, args.output)
                print(f"[{i:4d}/{len(tasks)}] {r.get('instance_id'):28s} "
                      f"feats={r.get('feature_label'):4s} "
                      f"nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')} "
                      f"status={r.get('status')}", flush=True)

    print(f"V4 Phase 1 done in {time.perf_counter() - t0:.0f}s "
          f"({len(rows)} rows).", flush=True)


if __name__ == '__main__':
    main()
