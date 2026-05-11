"""
V4 solver wrapper. Mirrors uc_hard_solver.solve_uc_with_settings but
dispatches to build_three_bin_v4 and forwards a `features` dict.

Returns the same result-dict shape as v3 plus F1..F4 columns.
"""

import os
import sys
import time

import highspy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from uc_formulations_v4 import build_three_bin_v4  # noqa: E402


def solve_uc_v4(inst, features=None,
                tight_minupdn=False,
                presolve='on',
                heuristics_effort=0.05,
                detect_symmetry=True,
                time_limit=120.0,
                gap_tol=0.001,
                verbose=False):
    """V4 solve: HiGHS with explicit feature toggles."""
    n_gen = inst['n_gen']
    horizon = inst['horizon']

    t0 = time.perf_counter()
    h, info = build_three_bin_v4(inst, features=features,
                                 tight_minupdn=tight_minupdn)
    build_time = time.perf_counter() - t0

    n_var = info['n_var_total']
    n_constr = info['n_constr_total']

    # LP relaxation for root-gap measurement
    h_lp = highspy.Highs(); h_lp.silent()
    lp_copy = h.getLp()
    lp_copy.integrality_ = [highspy.HighsVarType.kContinuous] * lp_copy.num_col_

    t0 = time.perf_counter()
    h_lp.passModel(lp_copy)
    h_lp.setOptionValue('output_flag', False)
    h_lp.run()
    lp_relax_time = time.perf_counter() - t0
    lp_relax_obj = float(h_lp.getInfo().objective_function_value)
    lp_iters = int(h_lp.getInfo().simplex_iteration_count)

    # MIP solve
    h.setOptionValue('time_limit', float(time_limit))
    h.setOptionValue('mip_rel_gap', gap_tol)
    h.setOptionValue('output_flag', verbose)
    h.setOptionValue('presolve', presolve)
    h.setOptionValue('mip_heuristic_effort', float(heuristics_effort))
    h.setOptionValue('mip_detect_symmetry', bool(detect_symmetry))

    t0 = time.perf_counter()
    h.run()
    mip_time = time.perf_counter() - t0

    mi = h.getInfo()
    mip_obj = float(mi.objective_function_value)
    dual_bound = float(mi.mip_dual_bound)
    mip_gap = float(mi.mip_gap)
    nodes = int(mi.mip_node_count)
    simp_iters = int(mi.simplex_iteration_count)
    status = h.modelStatusToString(h.getModelStatus())
    converged = ('optimal' in status.lower()) and (mip_gap <= gap_tol * 1.01)

    root_lp_gap_pct = (mip_obj - lp_relax_obj) / mip_obj * 100 if mip_obj > 0 else 0.0

    f = info['features']
    return {
        'n_gen': n_gen,
        'horizon': horizon,
        'n_var': n_var,
        'n_constr': n_constr,
        'n_var_binary': info['n_var_binary'],
        'build_time_s': build_time,
        'lp_relax_time_s': lp_relax_time,
        'lp_relax_obj': lp_relax_obj,
        'lp_relax_iters': lp_iters,
        'mip_total_time_s': mip_time,
        'mip_obj': mip_obj,
        'mip_dual_bound': dual_bound,
        'mip_gap': mip_gap,
        'mip_node_count': nodes,
        'mip_simplex_iters': simp_iters,
        'converged': converged,
        'status': status,
        'root_lp_gap_pct': root_lp_gap_pct,
        'presolve': presolve,
        'heuristics_effort': heuristics_effort,
        'detect_symmetry': detect_symmetry,
        'F1': int(f['F1']),
        'F2': int(f['F2']),
        'F3': int(f['F3']),
        'F4': int(f['F4']),
        'feature_label': ''.join(str(int(f[k])) for k in ('F1', 'F2', 'F3', 'F4')),
    }


if __name__ == '__main__':
    from itertools import product
    sys.path.insert(0, PARENT)
    from uc_hard_instance import generate_hard_uc, FLEET_RECIPES
    from instance_adapter import to_formulation_format
    import numpy as np

    hard = generate_hard_uc(FLEET_RECIPES['diverse_20'], horizon=8,
                            reserve_frac=0.15, plateau_factor=1.0, seed=42)
    inst = to_formulation_format(hard)
    inst['ramp_startup_limit'] = inst['P_min'] + 0.3 * (inst['P_max'] - inst['P_min'])
    inst['ramp_shutdown_limit'] = inst['P_min'] + 0.3 * (inst['P_max'] - inst['P_min'])
    inst['must_run'] = np.zeros(inst['n_gen'], dtype=bool)
    inst['must_run'][:3] = True
    inst['startup_categories'] = [
        [(1, 3, float(inst['cost_su'][i])),
         (4, 8, float(inst['cost_su'][i]) * 1.5),
         (9, 9999, float(inst['cost_su'][i]) * 2.5)]
        for i in range(inst['n_gen'])
    ]

    print(f"{'feats':6s} {'status':>10s} {'nodes':>6s} {'time':>7s} "
          f"{'gap%':>7s}")
    print('-' * 50)
    for combo in product([0, 1], repeat=4):
        feats = dict(zip(['F1', 'F2', 'F3', 'F4'], [bool(x) for x in combo]))
        r = solve_uc_v4(inst, features=feats, time_limit=30)
        print(f"{r['feature_label']:6s} {r['status']:>10s} "
              f"{r['mip_node_count']:6d} "
              f"{r['mip_total_time_s']:6.2f}s "
              f"{r['root_lp_gap_pct']:6.3f}%")
