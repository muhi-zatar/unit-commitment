"""
Phase 2: ablate HiGHS detect_symmetry on the top-decile hardest Phase 1 instances.

For each selected (N_g, k, T, seed) instance, re-solve twice:
  - detect_symmetry=False
  - detect_symmetry=True
and pair the results.

Writes phase2_results.csv.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from v3_common import (  # noqa: E402
    _highs_worker, checkpoint_csv, force_single_thread_env,
)


OUTPUT_CSV = os.path.join(THIS_DIR, 'phase2_results.csv')
PHASE1_CSV = os.path.join(THIS_DIR, 'phase1_results.csv')


def select_hard_instances(phase1_csv, top_frac=0.10, min_nodes=10):
    """Pick the top `top_frac` of Phase 1 runs by mip_node_count."""
    df = pd.read_csv(phase1_csv)
    df['mip_node_count'] = pd.to_numeric(df['mip_node_count'], errors='coerce')
    df = df.dropna(subset=['mip_node_count'])
    df = df[df['mip_node_count'] >= min_nodes]
    if len(df) == 0:
        return df
    n_pick = max(5, int(len(df) * top_frac))
    df = df.sort_values('mip_node_count', ascending=False).head(n_pick)
    return df[['N_g', 'k_identical', 'horizon', 'seed', 'mip_node_count']]


def build_tasks(selected, time_limit=300.0):
    tasks = []
    for _, row in selected.iterrows():
        for label, flag in (('sym_off', False), ('sym_on', True)):
            tasks.append({
                'N_g': int(row['N_g']),
                'k_identical': int(row['k_identical']),
                'horizon': int(row['horizon']),
                'seed': int(row['seed']),
                'formulation': 'three_bin',
                'detect_symmetry': flag,
                'time_limit': time_limit,
                'gap_tol': 0.001,
                'config_label': label,
            })
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--phase1-csv', default=PHASE1_CSV)
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--top-frac', type=float, default=0.10)
    parser.add_argument('--time-limit', type=float, default=300.0)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    force_single_thread_env()
    selected = select_hard_instances(args.phase1_csv, top_frac=args.top_frac)
    if args.quick:
        selected = selected.head(3)
    print(f"Phase 2: {len(selected)} hard instances selected from {args.phase1_csv}")
    if len(selected) == 0:
        print("No hard instances available — skipping Phase 2.")
        return
    print(selected.to_string(index=False))

    tasks = build_tasks(selected,
                        time_limit=30.0 if args.quick else args.time_limit)
    print(f"Phase 2: {len(tasks)} paired runs.")

    rows = []
    t0 = time.perf_counter()
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            r = _highs_worker(task)
            rows.append(r)
            checkpoint_csv(rows, args.output)
            print(f"[{i:3d}/{len(tasks)}] {r['instance_id']} "
                  f"{r['config_label']} nodes={r.get('mip_node_count')} "
                  f"time={r.get('mip_total_time_s')}", flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_highs_worker, tasks), 1):
                rows.append(r)
                checkpoint_csv(rows, args.output)
                print(f"[{i:3d}/{len(tasks)}] {r.get('instance_id')} "
                      f"{r.get('config_label')} nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')}", flush=True)

    print(f"Phase 2 done in {time.perf_counter() - t0:.0f}s")


if __name__ == '__main__':
    main()
