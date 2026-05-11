"""
V4 Phase 2: 2^4 factorial × small PGLib-UC subset.

Reads PGLib instances from --pglib-root (default: uc_experiment_v3/pglib-uc/).
Picks small instances (<= 50 thermals) up to --limit (default 8). Runs all
16 feature combos on each, horizon=24, time_limit=300s.

Output: uc_experiment_v4/phase2_v4_results.csv.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
V3_DIR = os.path.join(PARENT, 'uc_experiment_v3')
for p in (PARENT, V3_DIR, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from v4_common import _pglib_worker, checkpoint_csv, force_single_thread_env  # noqa: E402

OUTPUT_CSV = os.path.join(THIS_DIR, 'phase2_v4_results.csv')
DEFAULT_PGLIB_ROOT = os.path.join(V3_DIR, 'pglib-uc')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pglib-root', default=DEFAULT_PGLIB_ROOT)
    parser.add_argument('--datasets', nargs='+', default=['ca', 'ferc', 'rts_gmlc'])
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--time-limit', type=float, default=300.0)
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--limit', type=int, default=8,
                        help='Max PGLib instances (sorted by n_gen ascending).')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.pglib_root):
        print(f"ERROR: PGLib root not found at {args.pglib_root}")
        print(f"Run: python {V3_DIR}/fetch_pglib_uc.py --dest {args.pglib_root}")
        sys.exit(1)

    # Discover via the v3 helper, then keep only the smallest n_gen ones.
    from pglib_uc_adapter import discover_instances
    instances = discover_instances(args.pglib_root,
                                   datasets=tuple(args.datasets),
                                   max_size='medium')
    instances.sort(key=lambda x: x['n_gen'])
    instances = instances[:args.limit]
    if args.quick:
        instances = instances[:1]
    if not instances:
        print("No PGLib instances found.")
        sys.exit(0)

    print(f"V4 Phase 2 (pglib): {len(instances)} instances × 16 feature combos = "
          f"{len(instances) * 16} runs")
    for inst in instances:
        print(f"  {inst['dataset']}/{inst['name']:30s} "
              f"n_gen={inst['n_gen']:3d}  ({inst['size_tag']})")

    combos = ([(0, 0, 0, 0), (1, 1, 1, 1)] if args.quick
              else list(product([0, 1], repeat=4)))
    horizon = 12 if args.quick else args.horizon
    time_limit = 30.0 if args.quick else args.time_limit

    tasks = []
    for ins in instances:
        for combo in combos:
            feats = dict(zip(['F1', 'F2', 'F3', 'F4'], [bool(x) for x in combo]))
            tasks.append({
                'json_path': ins['path'],
                'name': ins['name'], 'dataset': ins['dataset'],
                'horizon': horizon,
                'features': feats,
                'time_limit': time_limit, 'gap_tol': 0.001,
            })

    force_single_thread_env()
    rows = []
    t0 = time.perf_counter()
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            r = _pglib_worker(task)
            rows.append(r); checkpoint_csv(rows, args.output)
            print(f"[{i:4d}/{len(tasks)}] {r.get('instance_id'):40s} "
                  f"feats={r.get('feature_label'):4s} "
                  f"nodes={r.get('mip_node_count')} "
                  f"time={r.get('mip_total_time_s')} "
                  f"status={r.get('status')}", flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_pglib_worker, tasks), 1):
                rows.append(r); checkpoint_csv(rows, args.output)
                print(f"[{i:4d}/{len(tasks)}] {r.get('instance_id'):40s} "
                      f"feats={r.get('feature_label'):4s} "
                      f"nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')} "
                      f"status={r.get('status')}", flush=True)

    print(f"V4 Phase 2 done in {time.perf_counter() - t0:.0f}s "
          f"({len(rows)} rows).", flush=True)


if __name__ == '__main__':
    main()
