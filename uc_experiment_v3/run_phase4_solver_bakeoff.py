"""
Phase 4: HiGHS vs SCIP bake-off on a Phase-1 subset.

Subset: N_g=24, T=24, k_identical in {1,4,8,16,24}, seeds=[42..46].
Each instance is solved by both solvers with default settings.

If pyscipopt is not installed, the SCIP runs are skipped and the CSV
contains HiGHS-only rows. This is the documented graceful fallback.

Writes phase4_results.csv.
"""

import argparse
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

from v3_common import (  # noqa: E402
    _highs_worker, checkpoint_csv, force_single_thread_env,
)


OUTPUT_CSV = os.path.join(THIS_DIR, 'phase4_results.csv')

SUBSET_K = [1, 4, 8, 16, 24]
SEEDS = [42, 43, 44, 45, 46]


def _scip_worker(task):
    """Top-level worker for SCIP; mirrors _highs_worker contract."""
    try:
        # Imports inside the child
        sys.path.insert(0, PARENT)
        sys.path.insert(0, THIS_DIR)
        from symmetry_sweep_instances import (
            make_symmetry_sweep_instance, instance_id,
        )
        from instance_adapter import to_formulation_format
        from uc_solver_scip import solve_uc_with_settings_scip, HAS_SCIP

        if not HAS_SCIP:
            raise RuntimeError("pyscipopt not installed in this worker")

        N_g = task['N_g']; k = task['k_identical']
        T = task['horizon']; seed = task['seed']

        hard = make_symmetry_sweep_instance(N_g, k, T, seed)
        inst = to_formulation_format(hard)

        r = solve_uc_with_settings_scip(
            inst, formulation=task.get('formulation', 'three_bin'),
            time_limit=task.get('time_limit', 300.0),
            gap_tol=task.get('gap_tol', 0.001),
        )
        r['instance_id'] = instance_id(N_g, k, T, seed)
        r['N_g'] = N_g; r['k_identical'] = k
        r['horizon'] = T; r['seed'] = seed
        r['config_label'] = task.get('config_label', 'default')
        r['solver'] = 'scip'
        r['error'] = ''
        return r
    except Exception as e:
        return {
            'instance_id': task.get('instance_id', ''),
            'N_g': task.get('N_g'),
            'k_identical': task.get('k_identical'),
            'horizon': task.get('horizon'),
            'seed': task.get('seed'),
            'config_label': task.get('config_label', 'default'),
            'solver': 'scip',
            'mip_total_time_s': None,
            'mip_node_count': None,
            'root_lp_gap_pct': None,
            'mip_simplex_iters': None,
            'status': 'ERROR',
            'converged': False,
            'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def build_tasks(time_limit=300.0):
    highs_tasks, scip_tasks = [], []
    for k in SUBSET_K:
        if k > 24:
            continue
        for seed in SEEDS:
            common = dict(N_g=24, k_identical=k, horizon=24, seed=seed,
                          formulation='three_bin',
                          time_limit=time_limit, gap_tol=0.001,
                          config_label='default')
            highs_tasks.append(common)
            scip_tasks.append(common)
    return highs_tasks, scip_tasks


def _check_scip():
    try:
        import pyscipopt  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--time-limit', type=float, default=300.0)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-scip', action='store_true')
    args = parser.parse_args()

    force_single_thread_env()
    have_scip = (not args.skip_scip) and _check_scip()
    if not have_scip and not args.skip_scip:
        print("WARNING: pyscipopt not available — running HiGHS only. "
              "Phase 4 ships partial.", flush=True)

    highs_tasks, scip_tasks = build_tasks(
        time_limit=30.0 if args.quick else args.time_limit)
    if args.quick:
        highs_tasks = highs_tasks[:3]
        scip_tasks = scip_tasks[:3]

    rows = []
    t0 = time.perf_counter()

    def _run(label, tasks, worker):
        if args.workers == 1:
            for i, task in enumerate(tasks, 1):
                r = worker(task)
                rows.append(r)
                checkpoint_csv(rows, args.output)
                print(f"[{label} {i:3d}/{len(tasks)}] {r['instance_id']} "
                      f"nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')}", flush=True)
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(args.workers) as pool:
                for i, r in enumerate(pool.imap_unordered(worker, tasks), 1):
                    rows.append(r)
                    checkpoint_csv(rows, args.output)
                    print(f"[{label} {i:3d}/{len(tasks)}] {r.get('instance_id')} "
                          f"nodes={r.get('mip_node_count')} "
                          f"time={r.get('mip_total_time_s')}", flush=True)

    _run('highs', highs_tasks, _highs_worker)
    if have_scip:
        _run('scip', scip_tasks, _scip_worker)

    print(f"Phase 4 done in {time.perf_counter() - t0:.0f}s "
          f"(scip_included={have_scip})")


if __name__ == '__main__':
    main()
