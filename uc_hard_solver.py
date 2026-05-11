"""
UC solver with controllable presolve / heuristics / symmetry options.

Mirrors uc_solver.solve_uc but accepts a settings dict that controls:
  - presolve         : 'on' / 'off'
  - heuristics       : 0.0 to 1.0 (effort)
  - detect_symmetry  : True / False  -- this is the big one for symmetric fleets
  - cuts             : 'on' / 'off' (approximated via mip_lp_age_limit)
"""

import time
import highspy

from uc_formulations import build_uc


def solve_uc_with_settings(inst, formulation='three_bin',
                            presolve='on',
                            heuristics_effort=0.05,
                            detect_symmetry=True,
                            time_limit=120.0,
                            gap_tol=0.001,
                            verbose=False):
    """Solve with explicit option control."""
    n_gen = inst['n_gen']
    horizon = inst['horizon']

    t0 = time.perf_counter()
    h, info = build_uc(inst, formulation=formulation)
    build_time = time.perf_counter() - t0

    n_var = info['n_var_total']
    n_constr = info['n_constr_total']

    # ---- LP relaxation for gap measurement ----
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

    # ---- MIP solve with the controlled settings ----
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

    return {
        'formulation': formulation,
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
        # Settings echoed for analysis
        'presolve': presolve,
        'heuristics_effort': heuristics_effort,
        'detect_symmetry': detect_symmetry,
    }


if __name__ == '__main__':
    from uc_hard_instance import generate_hard_uc, FLEET_RECIPES
    from instance_adapter import to_formulation_format

    print("=== Effect of solver settings on branching ===\n")
    recipe = FLEET_RECIPES['sym8_24']
    hard = generate_hard_uc(recipe, horizon=24, reserve_frac=0.20,
                            plateau_factor=0.0, seed=42)
    inst = to_formulation_format(hard)
    print(f"Recipe: sym8_24, n_gen={inst['n_gen']}, plateau=0 (perfectly identical units)")
    print(f"sym_score = {hard['symmetry_score']:.0f}\n")

    settings_grid = [
        ('default', dict(presolve='on', heuristics_effort=0.05, detect_symmetry=True)),
        ('no_symmetry', dict(presolve='on', heuristics_effort=0.05, detect_symmetry=False)),
        ('no_heuristics', dict(presolve='on', heuristics_effort=0.0, detect_symmetry=True)),
        ('no_presolve', dict(presolve='off', heuristics_effort=0.05, detect_symmetry=True)),
        ('all_off', dict(presolve='off', heuristics_effort=0.0, detect_symmetry=False)),
    ]

    print(f"{'setting':18s} | {'time':>8s} | {'nodes':>7s} | {'gap':>8s} | status")
    print('-' * 60)
    for name, settings in settings_grid:
        r = solve_uc_with_settings(inst, formulation='three_bin',
                                    time_limit=60, **settings)
        print(f"{name:18s} | {r['mip_total_time_s']:7.2f}s | "
              f"{r['mip_node_count']:7d} | {r['root_lp_gap_pct']:7.3f}% | {r['status']}")
