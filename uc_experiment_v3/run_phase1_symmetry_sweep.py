"""
Phase 1: continuous symmetry sweep.

Vary k_identical across {1,2,4,6,8,12,16,24} for N_g=24 and a wider grid for
N_g=48. Two horizons (12, 24), five seeds, three_bin formulation, HiGHS default.

Writes uc_experiment_v3/phase1_results.csv (checkpointed every run).
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

# Allow `python uc_experiment_v3/run_phase1_symmetry_sweep.py`
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from v3_common import (  # noqa: E402
    _highs_worker, checkpoint_csv, force_single_thread_env,
)


OUTPUT_CSV = os.path.join(THIS_DIR, 'phase1_results.csv')

K_GRID = {
    24: [1, 2, 4, 6, 8, 12, 16, 24],
    48: [1, 2, 4, 8, 12, 16, 24, 32, 48],
}
HORIZONS = [12, 24]
SEEDS = [42, 43, 44, 45, 46]
TIME_LIMIT = 300.0


def build_tasks(quick=False):
    tasks = []
    sizes = [24] if quick else [24, 48]
    seeds = [42] if quick else SEEDS
    horizons = [12] if quick else HORIZONS
    for N_g in sizes:
        ks = K_GRID[N_g][:3] if quick else K_GRID[N_g]
        for k in ks:
            for T in horizons:
                for seed in seeds:
                    tasks.append({
                        'N_g': N_g, 'k_identical': k, 'horizon': T, 'seed': seed,
                        'formulation': 'three_bin',
                        'time_limit': 30.0 if quick else TIME_LIMIT,
                        'gap_tol': 0.001,
                        'config_label': 'default',
                    })
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--quick', action='store_true',
                        help='Tiny subset for smoke-testing.')
    parser.add_argument('--output', default=OUTPUT_CSV)
    args = parser.parse_args()

    force_single_thread_env()
    tasks = build_tasks(quick=args.quick)
    print(f"Phase 1: {len(tasks)} runs, {args.workers} workers, "
          f"output={args.output}", flush=True)

    rows = []
    t0 = time.perf_counter()
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            r = _highs_worker(task)
            rows.append(r)
            checkpoint_csv(rows, args.output)
            elapsed = time.perf_counter() - t0
            print(f"[{i:3d}/{len(tasks)}] {r.get('instance_id','?')} "
                  f"nodes={r.get('mip_node_count')} "
                  f"time={r.get('mip_total_time_s')} "
                  f"status={r.get('status')} "
                  f"(elapsed {elapsed:.0f}s)", flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_highs_worker, tasks), 1):
                rows.append(r)
                checkpoint_csv(rows, args.output)
                elapsed = time.perf_counter() - t0
                print(f"[{i:3d}/{len(tasks)}] {r.get('instance_id','?')} "
                      f"nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')} "
                      f"status={r.get('status')} "
                      f"(elapsed {elapsed:.0f}s)", flush=True)

    print(f"Phase 1 done: {len(rows)} rows, "
          f"{time.perf_counter() - t0:.0f}s total.", flush=True)


if __name__ == '__main__':
    main()
