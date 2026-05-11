"""
V5 worker plumbing. Re-uses v3's checkpoint_csv / force_single_thread_env.
"""

import os
import sys
import time
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
V3_DIR = os.path.join(PARENT, 'uc_experiment_v3')
for p in (PARENT, V3_DIR, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from v3_common import checkpoint_csv, force_single_thread_env  # noqa: F401, E402


def _v5_worker(task):
    """
    Top-level multiprocessing worker.

    task keys:
        network, solver, horizon, seed, time_limit, gap_tol,
        rts_path (optional, only for rts_gmlc)
    """
    try:
        sys.path.insert(0, THIS_DIR)
        from egret_loader import NETWORK_LOADERS, build_pyomo_scuc, network_size
        from pyomo_solver_bridge import solve_pyomo_uc

        network = task['network']
        horizon = task.get('horizon', 24)
        seed = task.get('seed', 42)

        loader = NETWORK_LOADERS[network]
        if network == 'rts_gmlc' and task.get('rts_path'):
            md = loader(horizon=horizon, seed=seed,
                        rts_path=task['rts_path'])
        else:
            md = loader(horizon=horizon, seed=seed)
        n_bus, n_branch, n_gen, T = network_size(md)

        t0 = time.perf_counter()
        model = build_pyomo_scuc(md)
        build_time = time.perf_counter() - t0

        r = solve_pyomo_uc(
            model,
            solver=task['solver'],
            time_limit=task.get('time_limit', 600.0),
            gap_tol=task.get('gap_tol', 0.001),
        )

        r['network'] = network
        r['n_bus'] = n_bus; r['n_branch'] = n_branch
        r['n_gen'] = n_gen; r['horizon'] = T
        r['seed'] = seed
        r['build_time_s'] = build_time
        r['instance_id'] = f"{network}_T{T}_s{seed}"
        r['error'] = ''
        return r
    except Exception as e:
        return {
            'network': task.get('network'),
            'solver': task.get('solver'),
            'horizon': task.get('horizon'),
            'seed': task.get('seed'),
            'instance_id': f"{task.get('network','?')}_T{task.get('horizon')}_s{task.get('seed')}",
            'mip_total_time_s': None,
            'mip_node_count': None,
            'mip_obj': None,
            'mip_gap': None,
            'status': 'ERROR',
            'converged': False,
            'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
