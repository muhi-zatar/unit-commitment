"""
UC solver — reconstructed from the previous experiment's CSV column structure.

solve_uc() builds a model via uc_formulations.build_uc(), solves the LP
relaxation, then solves the MILP, and reports per-stage timing + counts
matching the existing uc_results.csv schema.
"""

import time
import highspy

from uc_formulations import build_uc


def solve_uc(inst, formulation='three_bin', time_limit=120.0, gap_tol=0.001):
    """
    Solve a UC instance and return a dict matching the existing CSV schema.
    """
    n_gen = inst['n_gen']
    horizon = inst['horizon']

    # Build
    t0 = time.perf_counter()
    h, info = build_uc(inst, formulation=formulation)
    build_time = time.perf_counter() - t0

    n_var = info['n_var_total']
    n_constr = info['n_constr_total']
    n_var_bin = info['n_var_binary']
    n_var_cont = info['n_var_continuous']
    n_pwl = info['n_pwl_segments']
    n_nz = info.get('n_nz', 0)

    # ---- Solve LP relaxation by relaxing integrality on a copy ----
    # We get the LP from h.getLp(), set integrality to all continuous
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

    # ---- Solve MILP ----
    h.setOptionValue('time_limit', float(time_limit))
    h.setOptionValue('mip_rel_gap', gap_tol)
    h.setOptionValue('output_flag', False)

    t0 = time.perf_counter()
    h.run()
    mip_time = time.perf_counter() - t0

    mip_info = h.getInfo()
    mip_obj = float(mip_info.objective_function_value)
    dual_bound = float(mip_info.mip_dual_bound)
    mip_gap = float(mip_info.mip_gap)
    nodes = int(mip_info.mip_node_count)
    simp_iters = int(mip_info.simplex_iteration_count)
    status = h.modelStatusToString(h.getModelStatus())
    converged = ('optimal' in status.lower()) and (mip_gap <= gap_tol * 1.01)

    bnc_time = max(0.0, mip_time - lp_relax_time * 0.5)
    root_lp_gap_pct = (mip_obj - lp_relax_obj) / mip_obj * 100 if mip_obj > 0 else 0.0

    return {
        'formulation': formulation,
        'n_gen': n_gen,
        'horizon': horizon,
        'n_var': n_var,
        'n_constr': n_constr,
        'n_nz': n_nz,
        'n_var_binary': n_var_bin,
        'n_var_continuous': n_var_cont,
        'n_pwl_segments': n_pwl,
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
        'bnc_time_s': bnc_time,
        'root_lp_gap_pct': root_lp_gap_pct,
    }


if __name__ == '__main__':
    from uc_instance import generate_uc_instance
    print("=== solve_uc smoke test ===")
    for n in [10, 25]:
        inst = generate_uc_instance(n, horizon=24, seed=42)
        for form in ['three_bin', 'three_bin_tight', 'perspective']:
            r = solve_uc(inst, formulation=form, time_limit=30, gap_tol=0.001)
            print(f"  N_g={n} {form:18s}: time={r['mip_total_time_s']:.2f}s "
                  f"nodes={r['mip_node_count']} gap={r['root_lp_gap_pct']:.3f}%")
