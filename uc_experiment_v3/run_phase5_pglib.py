"""
Phase 5: HiGHS vs SCIP on PGLib-UC small/medium instances.

Prerequisite: a local PGLib-UC checkout (run `fetch_pglib_uc.py` first, or
pass --pglib-root to point at an existing clone).

Each discovered instance is solved by both solvers with default settings.
Time horizon is optionally truncated to keep runtime in budget.

Output: phase5_results.csv (atomic checkpoint after each run).
"""

import argparse
import json
import os
import sys
import time
import traceback
import multiprocessing as mp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from v3_common import checkpoint_csv, force_single_thread_env  # noqa: E402


OUTPUT_CSV = os.path.join(THIS_DIR, 'phase5_results.csv')
DEFAULT_PGLIB_ROOT = os.path.join(THIS_DIR, 'pglib-uc')


def _pglib_highs_worker(task):
    try:
        sys.path.insert(0, PARENT); sys.path.insert(0, THIS_DIR)
        from pglib_uc_adapter import load_pglib_instance
        from uc_hard_solver import solve_uc_with_settings

        inst = load_pglib_instance(task['json_path'],
                                   horizon=task.get('horizon'),
                                   net_renewables=task.get('net_renewables', False))
        r = solve_uc_with_settings(
            inst, formulation='three_bin',
            time_limit=task.get('time_limit', 600.0),
            gap_tol=task.get('gap_tol', 0.001),
        )
        r['instance_id'] = f"{task['dataset']}/{task['name']}_T{inst['horizon']}"
        r['pglib_name'] = task['name']
        r['dataset'] = task['dataset']
        r['size_tag'] = task['size_tag']
        r['horizon'] = inst['horizon']
        r['n_gen'] = inst['n_gen']
        r['solver'] = 'highs'
        r['error'] = ''
        return r
    except Exception as e:
        return _err_row(task, 'highs', e)


def _pglib_scip_worker(task):
    try:
        sys.path.insert(0, PARENT); sys.path.insert(0, THIS_DIR)
        from pglib_uc_adapter import load_pglib_instance
        from uc_solver_scip import solve_uc_with_settings_scip, HAS_SCIP
        if not HAS_SCIP:
            raise RuntimeError("pyscipopt not installed in this worker")

        inst = load_pglib_instance(task['json_path'],
                                   horizon=task.get('horizon'),
                                   net_renewables=task.get('net_renewables', False))
        r = solve_uc_with_settings_scip(
            inst, formulation='three_bin',
            time_limit=task.get('time_limit', 600.0),
            gap_tol=task.get('gap_tol', 0.001),
        )
        r['instance_id'] = f"{task['dataset']}/{task['name']}_T{inst['horizon']}"
        r['pglib_name'] = task['name']
        r['dataset'] = task['dataset']
        r['size_tag'] = task['size_tag']
        r['horizon'] = inst['horizon']
        r['n_gen'] = inst['n_gen']
        r['solver'] = 'scip'
        r['error'] = ''
        return r
    except Exception as e:
        return _err_row(task, 'scip', e)


def _err_row(task, solver, e):
    return {
        'instance_id': f"{task.get('dataset')}/{task.get('name')}",
        'pglib_name': task.get('name'),
        'dataset': task.get('dataset'),
        'size_tag': task.get('size_tag'),
        'horizon': task.get('horizon'),
        'n_gen': task.get('n_gen'),
        'solver': solver,
        'mip_total_time_s': None,
        'mip_node_count': None,
        'status': 'ERROR',
        'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pglib-root', default=DEFAULT_PGLIB_ROOT)
    parser.add_argument('--datasets', nargs='+', default=['ca', 'ferc', 'rts_gmlc'])
    parser.add_argument('--max-size', choices=['small', 'medium', 'large'],
                        default='medium')
    parser.add_argument('--horizon', type=int, default=24,
                        help='Truncate to this many hours (None to keep full).')
    parser.add_argument('--time-limit', type=float, default=600.0)
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--net-renewables', action='store_true')
    parser.add_argument('--skip-scip', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only solve the first N discovered instances.')
    args = parser.parse_args()

    if not os.path.isdir(args.pglib_root):
        print(f"ERROR: PGLib-UC root not found at {args.pglib_root}")
        print(f"Run: python {os.path.relpath(os.path.join(THIS_DIR, 'fetch_pglib_uc.py'))}")
        sys.exit(1)

    from pglib_uc_adapter import discover_instances
    instances = discover_instances(args.pglib_root,
                                   datasets=tuple(args.datasets),
                                   max_size=args.max_size)
    if args.quick:
        instances = instances[:1]
    elif args.limit:
        instances = instances[:args.limit]

    if not instances:
        print("No PGLib instances discovered.")
        sys.exit(0)
    print(f"Phase 5: {len(instances)} instances "
          f"(datasets={args.datasets}, max_size={args.max_size}, "
          f"horizon={args.horizon})")
    for inst in instances:
        print(f"  {inst['dataset']}/{inst['name']:30s} "
              f"n_gen={inst['n_gen']:3d}  T={inst['time_periods']:3d}  "
              f"({inst['size_tag']})")

    force_single_thread_env()

    common = {
        'horizon': args.horizon,
        'time_limit': 30.0 if args.quick else args.time_limit,
        'net_renewables': args.net_renewables,
    }
    tasks = [{**inst,
              'json_path': inst['path'],
              **common}
             for inst in instances]

    have_scip = (not args.skip_scip) and _has_scip()
    if not have_scip and not args.skip_scip:
        print("WARNING: pyscipopt not available — running HiGHS only.", flush=True)

    rows = []
    t0 = time.perf_counter()

    def _run(label, worker):
        if args.workers == 1:
            for i, task in enumerate(tasks, 1):
                r = worker(task)
                rows.append(r)
                checkpoint_csv(rows, args.output)
                print(f"[{label} {i:3d}/{len(tasks)}] {r['instance_id']} "
                      f"nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')} "
                      f"status={r.get('status')}", flush=True)
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(args.workers) as pool:
                for i, r in enumerate(pool.imap_unordered(worker, tasks), 1):
                    rows.append(r)
                    checkpoint_csv(rows, args.output)
                    print(f"[{label} {i:3d}/{len(tasks)}] {r.get('instance_id')} "
                          f"nodes={r.get('mip_node_count')} "
                          f"time={r.get('mip_total_time_s')} "
                          f"status={r.get('status')}", flush=True)

    _run('highs', _pglib_highs_worker)
    if have_scip:
        _run('scip', _pglib_scip_worker)

    print(f"Phase 5 done in {time.perf_counter() - t0:.0f}s "
          f"(scip_included={have_scip})")


def _has_scip():
    try:
        import pyscipopt  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == '__main__':
    main()
