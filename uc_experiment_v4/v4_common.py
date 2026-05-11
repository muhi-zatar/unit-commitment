"""
Shared plumbing for V4 drivers — multiprocessing workers that build the
instance, apply the feature toggle, and solve. Imports the v3 plumbing
(checkpoint_csv, force_single_thread_env) so we don't duplicate.
"""

import os
import sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
V3_DIR = os.path.join(PARENT, 'uc_experiment_v3')
if V3_DIR not in sys.path:
    sys.path.insert(0, V3_DIR)

# Re-export v3 utilities so v4 drivers don't have to know about v3 paths
from v3_common import checkpoint_csv, force_single_thread_env  # noqa: F401, E402


def _synth_worker(task):
    """Worker: build V4 synthetic instance, solve with feature toggle."""
    try:
        sys.path.insert(0, PARENT); sys.path.insert(0, THIS_DIR); sys.path.insert(0, V3_DIR)
        from uc_experiment_v4.symmetry_sweep_instances_v4 import (
            make_symmetry_sweep_instance_v4,
        )
        from uc_experiment_v4.uc_hard_solver_v4 import solve_uc_v4

        inst = make_symmetry_sweep_instance_v4(
            N_g=task['N_g'], k_identical=task['k_identical'],
            horizon=task['horizon'], seed=task['seed'])
        r = solve_uc_v4(inst,
                        features=task['features'],
                        time_limit=task.get('time_limit', 300.0),
                        gap_tol=task.get('gap_tol', 0.001))
        r['instance_set'] = 'synth'
        r['instance_id'] = f"Ng{task['N_g']}_k{task['k_identical']}_T{task['horizon']}_s{task['seed']}"
        r['N_g'] = task['N_g']; r['k_identical'] = task['k_identical']
        r['horizon'] = task['horizon']; r['seed'] = task['seed']
        r['error'] = ''
        return r
    except Exception as e:
        return _err(task, 'synth', e)


def _pglib_worker(task):
    """Worker: load PGLib V4 instance, solve with feature toggle."""
    try:
        sys.path.insert(0, PARENT); sys.path.insert(0, THIS_DIR); sys.path.insert(0, V3_DIR)
        from uc_experiment_v4.pglib_uc_adapter_v4 import load_pglib_instance_v4
        from uc_experiment_v4.uc_hard_solver_v4 import solve_uc_v4

        inst = load_pglib_instance_v4(task['json_path'],
                                       horizon=task['horizon'])
        r = solve_uc_v4(inst,
                        features=task['features'],
                        time_limit=task.get('time_limit', 300.0),
                        gap_tol=task.get('gap_tol', 0.001))
        r['instance_set'] = 'pglib'
        r['instance_id'] = f"{task['dataset']}/{task['name']}_T{task['horizon']}"
        r['pglib_name'] = task['name']
        r['dataset'] = task['dataset']
        r['horizon'] = task['horizon']
        r['seed'] = task.get('seed', 0)
        r['error'] = ''
        return r
    except Exception as e:
        return _err(task, 'pglib', e)


def _err(task, kind, e):
    return {
        'instance_set': kind,
        'instance_id': task.get('instance_id', ''),
        'N_g': task.get('N_g'),
        'k_identical': task.get('k_identical'),
        'horizon': task.get('horizon'),
        'seed': task.get('seed'),
        'feature_label': ''.join(str(int(task['features'][k]))
                                 for k in ('F1', 'F2', 'F3', 'F4')),
        'F1': int(task['features']['F1']),
        'F2': int(task['features']['F2']),
        'F3': int(task['features']['F3']),
        'F4': int(task['features']['F4']),
        'mip_total_time_s': None,
        'mip_node_count': None,
        'root_lp_gap_pct': None,
        'status': 'ERROR',
        'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
    }
