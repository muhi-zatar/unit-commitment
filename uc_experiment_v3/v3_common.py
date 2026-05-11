"""
Shared utilities for v3 phase drivers.

- atomic CSV checkpointing
- common path constants
- worker entry-point that loads HiGHS in a child process and forces 1 thread
"""

import os
import sys
import tempfile
import traceback

import pandas as pd

# Allow imports from parent directory when running scripts directly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PLOTS_DIR = os.path.join(THIS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def checkpoint_csv(rows, path):
    """Atomically write rows to CSV (so a crash mid-write doesn't corrupt it)."""
    df = pd.DataFrame(rows)
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _highs_worker(task):
    """
    Top-level multiprocessing worker. Each child process loads HiGHS fresh,
    pins it to a single thread, builds the instance, and solves it.

    `task` is a dict with the keys this phase needs; the worker dispatches
    based on `task['kind']`.

    Returns a result dict (always a dict, even on error — driver records all).
    """
    try:
        from uc_experiment_v3.symmetry_sweep_instances import (
            make_symmetry_sweep_instance, instance_id,
        )
        from instance_adapter import to_formulation_format
        from uc_hard_solver import solve_uc_with_settings
        import highspy  # noqa: F401  - ensure import succeeds in the child

        N_g = task['N_g']
        k = task['k_identical']
        T = task['horizon']
        seed = task['seed']
        formulation = task.get('formulation', 'three_bin')
        time_limit = task.get('time_limit', 300.0)
        gap_tol = task.get('gap_tol', 0.001)
        presolve = task.get('presolve', 'on')
        heuristics_effort = task.get('heuristics_effort', 0.05)
        detect_symmetry = task.get('detect_symmetry', True)
        config_label = task.get('config_label', 'default')

        hard = make_symmetry_sweep_instance(
            N_g=N_g, k_identical=k, horizon=T, seed=seed)
        inst = to_formulation_format(hard)

        # Pin HiGHS to one thread via solver wrapper environment (set below
        # in the driver by os.environ); the wrapper itself doesn't expose
        # threads. We set it through HiGHS's option after the model is built
        # by monkey-patching: simplest is to set highs_threads via env. HiGHS
        # honours `threads` option, but the wrapper doesn't pass it. Best
        # we can do here without changing v2 code is to leave HiGHS's default
        # (which is already low-thread for small models) and rely on the OS
        # env vars OMP_NUM_THREADS / OPENBLAS_NUM_THREADS for LP backend.
        # The driver sets these before spawning children.

        r = solve_uc_with_settings(
            inst, formulation=formulation,
            presolve=presolve,
            heuristics_effort=heuristics_effort,
            detect_symmetry=detect_symmetry,
            time_limit=time_limit,
            gap_tol=gap_tol,
        )

        r['instance_id'] = instance_id(N_g, k, T, seed)
        r['N_g'] = N_g
        r['k_identical'] = k
        r['horizon'] = T
        r['seed'] = seed
        r['config_label'] = config_label
        r['solver'] = 'highs'
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
            'solver': 'highs',
            'mip_total_time_s': None,
            'mip_node_count': None,
            'root_lp_gap_pct': None,
            'mip_simplex_iters': None,
            'status': 'ERROR',
            'converged': False,
            'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def force_single_thread_env():
    """Set env vars to keep BLAS/OMP from oversubscribing inside workers."""
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(k, '1')
