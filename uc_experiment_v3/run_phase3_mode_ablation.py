"""
Phase 3: solver-mode ablation on one mid-range instance.

Configs:
  default      : presolve=on,  heur=0.05, sym=on
  no_presolve  : presolve=off, heur=0.05, sym=on
  no_heuristics: presolve=on,  heur=0.00, sym=on
  no_symmetry  : presolve=on,  heur=0.05, sym=off
  raw          : presolve=off, heur=0.00, sym=off

Instance: N_g=24, k_identical=8, T=24, seeds=[42..46]. Time limit 600s.

Writes phase3_results.csv.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from v3_common import (  # noqa: E402
    _highs_worker, checkpoint_csv, force_single_thread_env,
)


OUTPUT_CSV = os.path.join(THIS_DIR, 'phase3_results.csv')

CONFIGS = [
    ('default',       dict(presolve='on',  heuristics_effort=0.05, detect_symmetry=True)),
    ('no_presolve',   dict(presolve='off', heuristics_effort=0.05, detect_symmetry=True)),
    ('no_heuristics', dict(presolve='on',  heuristics_effort=0.0,  detect_symmetry=True)),
    ('no_symmetry',   dict(presolve='on',  heuristics_effort=0.05, detect_symmetry=False)),
    ('raw',           dict(presolve='off', heuristics_effort=0.0,  detect_symmetry=False)),
]

INSTANCE = dict(N_g=24, k_identical=8, horizon=24)
SEEDS = [42, 43, 44, 45, 46]


def build_tasks(time_limit=600.0, seeds=None):
    seeds = seeds or SEEDS
    tasks = []
    for seed in seeds:
        for label, cfg in CONFIGS:
            tasks.append({
                'N_g': INSTANCE['N_g'],
                'k_identical': INSTANCE['k_identical'],
                'horizon': INSTANCE['horizon'],
                'seed': seed,
                'formulation': 'three_bin',
                'time_limit': time_limit,
                'gap_tol': 0.001,
                'config_label': label,
                **cfg,
            })
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--time-limit', type=float, default=600.0)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    force_single_thread_env()
    tasks = build_tasks(
        time_limit=30.0 if args.quick else args.time_limit,
        seeds=[42] if args.quick else SEEDS,
    )
    print(f"Phase 3: {len(tasks)} runs over {len(CONFIGS)} configs.")

    rows = []
    t0 = time.perf_counter()
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            r = _highs_worker(task)
            rows.append(r)
            checkpoint_csv(rows, args.output)
            print(f"[{i:3d}/{len(tasks)}] {r['instance_id']} "
                  f"{r['config_label']:<14} nodes={r.get('mip_node_count')} "
                  f"time={r.get('mip_total_time_s')} "
                  f"status={r.get('status')}", flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_highs_worker, tasks), 1):
                rows.append(r)
                checkpoint_csv(rows, args.output)
                print(f"[{i:3d}/{len(tasks)}] {r.get('instance_id')} "
                      f"{r.get('config_label','?'):<14} "
                      f"nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')} "
                      f"status={r.get('status')}", flush=True)

    print(f"Phase 3 done in {time.perf_counter() - t0:.0f}s")


if __name__ == '__main__':
    main()
