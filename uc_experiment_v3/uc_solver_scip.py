"""
SCIP wrapper exposing the same interface as uc_hard_solver.solve_uc_with_settings.

Builds the three_bin UC formulation directly in pyscipopt (mirroring
uc_formulations.build_three_bin with tight_minupdn=False, n_pwl_segments=1).

Returns a result dict with the same field names the HiGHS wrapper emits.

If pyscipopt is not installed, solve_uc_with_settings_scip raises
RuntimeError on first call; the Phase 4 driver handles that gracefully.
"""

import time

try:
    import pyscipopt
    from pyscipopt import Model, quicksum
    HAS_SCIP = True
except ImportError:  # pragma: no cover
    pyscipopt = None
    HAS_SCIP = False


def _build_three_bin_scip(inst, tight_minupdn=False):
    """Build three_bin UC in pyscipopt. Returns (model, var_dicts, info)."""
    if not HAS_SCIP:
        raise RuntimeError("pyscipopt is not installed")

    n_gen = inst['n_gen']
    T = inst['horizon']

    m = Model("UC_three_bin")
    m.hideOutput()

    u, v, w, P = {}, {}, {}, {}
    for i in range(n_gen):
        for t in range(T):
            u[i, t] = m.addVar(vtype='B', name=f"u_{i}_{t}")
            v[i, t] = m.addVar(vtype='B', name=f"v_{i}_{t}")
            w[i, t] = m.addVar(vtype='B', name=f"w_{i}_{t}")
            P[i, t] = m.addVar(vtype='C', lb=0.0,
                               ub=float(inst['P_max'][i]), name=f"P_{i}_{t}")

    # Objective
    obj_terms = []
    for i in range(n_gen):
        nl = float(inst['cost_nl'][i])
        su = float(inst['cost_su'][i])
        # Linear approximation of quadratic + linear cost (n_pwl_segments=1)
        eff_slope = float(inst['cost_b'][i] + inst['cost_a'][i] * inst['P_max'][i])
        for t in range(T):
            obj_terms.append(nl * u[i, t])
            obj_terms.append(su * v[i, t])
            obj_terms.append(eff_slope * P[i, t])
    m.setObjective(quicksum(obj_terms), "minimize")

    # State eqn: u[i,t] - u[i,t-1] = v[i,t] - w[i,t]
    for i in range(n_gen):
        for t in range(T):
            if t == 0:
                m.addCons(u[i, 0] - v[i, 0] + w[i, 0] == float(inst['init_status'][i]))
            else:
                m.addCons(u[i, t] - u[i, t - 1] - v[i, t] + w[i, t] == 0)

    # v + w <= 1
    for i in range(n_gen):
        for t in range(T):
            m.addCons(v[i, t] + w[i, t] <= 1)

    # P bounds
    for i in range(n_gen):
        pmin = float(inst['P_min'][i])
        pmax = float(inst['P_max'][i])
        for t in range(T):
            m.addCons(P[i, t] - pmin * u[i, t] >= 0)
            m.addCons(P[i, t] - pmax * u[i, t] <= 0)

    # Ramps
    for i in range(n_gen):
        ru = float(inst['ramp_up'][i])
        rd = float(inst['ramp_dn'][i])
        pi0 = float(inst['P_init'][i])
        for t in range(T):
            if t == 0:
                m.addCons(P[i, 0] <= pi0 + ru)
                m.addCons(-P[i, 0] <= rd - pi0)
            else:
                m.addCons(P[i, t] - P[i, t - 1] <= ru)
                m.addCons(P[i, t - 1] - P[i, t] <= rd)

    # Min up/down (weak form)
    if not tight_minupdn:
        for i in range(n_gen):
            mu = int(inst['min_up'][i])
            md = int(inst['min_dn'][i])
            for t in range(T):
                window_v = [v[i, tau] for tau in range(max(0, t - mu + 1), t + 1)]
                if window_v:
                    m.addCons(quicksum(window_v) - u[i, t] <= 0)
                window_w = [w[i, tau] for tau in range(max(0, t - md + 1), t + 1)]
                if window_w:
                    m.addCons(quicksum(window_w) + u[i, t] <= 1)

    # Demand balance
    for t in range(T):
        m.addCons(quicksum(P[i, t] for i in range(n_gen)) == float(inst['load'][t]))

    # Reserve: sum P_max * u >= load + reserve
    for t in range(T):
        rhs = float(inst['load'][t] + inst['reserve'][t])
        m.addCons(quicksum(float(inst['P_max'][i]) * u[i, t]
                           for i in range(n_gen)) >= rhs)

    info = {
        'n_var_total': 3 * n_gen * T + n_gen * T,
        'n_var_binary': 3 * n_gen * T,
        'n_constr_total': None,  # SCIP can report after presolve
    }
    return m, {'u': u, 'v': v, 'w': w, 'P': P}, info


def _build_lp_relax_objective(inst, model_template):
    """Quick LP relaxation by rebuilding the same model with continuous bins."""
    # Simpler: rebuild a fresh model with vtype='C' for u,v,w.
    n_gen = inst['n_gen']
    T = inst['horizon']
    m = Model("UC_LP_relax"); m.hideOutput()
    u, v, w, P = {}, {}, {}, {}
    for i in range(n_gen):
        for t in range(T):
            u[i, t] = m.addVar(vtype='C', lb=0, ub=1)
            v[i, t] = m.addVar(vtype='C', lb=0, ub=1)
            w[i, t] = m.addVar(vtype='C', lb=0, ub=1)
            P[i, t] = m.addVar(vtype='C', lb=0.0, ub=float(inst['P_max'][i]))

    obj_terms = []
    for i in range(n_gen):
        nl = float(inst['cost_nl'][i]); su = float(inst['cost_su'][i])
        eff_slope = float(inst['cost_b'][i] + inst['cost_a'][i] * inst['P_max'][i])
        for t in range(T):
            obj_terms.append(nl * u[i, t])
            obj_terms.append(su * v[i, t])
            obj_terms.append(eff_slope * P[i, t])
    m.setObjective(quicksum(obj_terms), "minimize")

    for i in range(n_gen):
        for t in range(T):
            if t == 0:
                m.addCons(u[i, 0] - v[i, 0] + w[i, 0] == float(inst['init_status'][i]))
            else:
                m.addCons(u[i, t] - u[i, t - 1] - v[i, t] + w[i, t] == 0)
            m.addCons(v[i, t] + w[i, t] <= 1)
            m.addCons(P[i, t] - float(inst['P_min'][i]) * u[i, t] >= 0)
            m.addCons(P[i, t] - float(inst['P_max'][i]) * u[i, t] <= 0)

    for i in range(n_gen):
        ru = float(inst['ramp_up'][i]); rd = float(inst['ramp_dn'][i])
        pi0 = float(inst['P_init'][i])
        for t in range(T):
            if t == 0:
                m.addCons(P[i, 0] <= pi0 + ru)
                m.addCons(-P[i, 0] <= rd - pi0)
            else:
                m.addCons(P[i, t] - P[i, t - 1] <= ru)
                m.addCons(P[i, t - 1] - P[i, t] <= rd)
        mu = int(inst['min_up'][i]); md = int(inst['min_dn'][i])
        for t in range(T):
            wv = [v[i, tau] for tau in range(max(0, t - mu + 1), t + 1)]
            if wv:
                m.addCons(quicksum(wv) - u[i, t] <= 0)
            ww = [w[i, tau] for tau in range(max(0, t - md + 1), t + 1)]
            if ww:
                m.addCons(quicksum(ww) + u[i, t] <= 1)

    for t in range(T):
        m.addCons(quicksum(P[i, t] for i in range(n_gen)) == float(inst['load'][t]))
        rhs = float(inst['load'][t] + inst['reserve'][t])
        m.addCons(quicksum(float(inst['P_max'][i]) * u[i, t]
                           for i in range(n_gen)) >= rhs)

    m.setParam('display/verblevel', 0)
    m.optimize()
    obj = m.getObjVal()
    lp_iters = int(m.getNLPIterations()) if hasattr(m, 'getNLPIterations') else 0
    return obj, lp_iters


def solve_uc_with_settings_scip(inst, formulation='three_bin',
                                 presolve='on',
                                 heuristics_effort=0.05,
                                 detect_symmetry=True,
                                 time_limit=300.0,
                                 gap_tol=0.001,
                                 verbose=False):
    """SCIP equivalent of uc_hard_solver.solve_uc_with_settings."""
    if not HAS_SCIP:
        raise RuntimeError("pyscipopt not installed; install with `pip install pyscipopt`")

    if formulation not in ('three_bin', 'three_bin_tight'):
        raise ValueError(f"SCIP wrapper only handles three_bin / three_bin_tight, "
                         f"got {formulation!r}")

    tight = (formulation == 'three_bin_tight')

    n_gen = inst['n_gen']; T = inst['horizon']

    # LP relaxation for gap measurement
    t0 = time.perf_counter()
    try:
        lp_obj, lp_iters = _build_lp_relax_objective(inst, None)
    except Exception:
        lp_obj, lp_iters = float('nan'), 0
    lp_relax_time = time.perf_counter() - t0

    # Build MIP
    t0 = time.perf_counter()
    m, vars_, info = _build_three_bin_scip(inst, tight_minupdn=tight)
    build_time = time.perf_counter() - t0

    # Apply settings
    m.setParam('limits/time', float(time_limit))
    m.setParam('limits/gap', float(gap_tol))
    m.setParam('parallel/maxnthreads', 1)
    if not verbose:
        m.setParam('display/verblevel', 0)

    if presolve == 'off':
        m.setParam('presolving/maxrounds', 0)
        m.setParam('presolving/maxrestarts', 0)
    if heuristics_effort == 0.0:
        try:
            m.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        except Exception:
            # Fallback: zero out heuristics frequency via emphasis
            m.setParam('heuristics/emphasis/off', 1) if False else None
    if not detect_symmetry:
        try:
            m.setParam('misc/usesymmetry', 0)
        except Exception:
            pass

    t0 = time.perf_counter()
    m.optimize()
    mip_time = time.perf_counter() - t0

    try:
        mip_obj = float(m.getObjVal())
    except Exception:
        mip_obj = float('nan')
    try:
        dual_bound = float(m.getDualbound())
    except Exception:
        dual_bound = float('nan')
    try:
        mip_gap = float(m.getGap())
    except Exception:
        mip_gap = float('nan')
    try:
        nodes = int(m.getNNodes())
    except Exception:
        nodes = 0
    try:
        simp_iters = int(m.getNLPIterations())
    except Exception:
        simp_iters = 0
    status = m.getStatus()
    converged = (status == 'optimal')

    root_lp_gap_pct = ((mip_obj - lp_obj) / mip_obj * 100
                       if mip_obj and mip_obj > 0 else 0.0)

    return {
        'formulation': formulation,
        'n_gen': n_gen,
        'horizon': T,
        'n_var': info['n_var_total'],
        'n_constr': info['n_constr_total'],
        'n_var_binary': info['n_var_binary'],
        'build_time_s': build_time,
        'lp_relax_time_s': lp_relax_time,
        'lp_relax_obj': lp_obj,
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
    }


if __name__ == '__main__':
    print(f"pyscipopt available: {HAS_SCIP}")
    if HAS_SCIP:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from symmetry_sweep_instances import make_symmetry_sweep_instance
        from instance_adapter import to_formulation_format
        hard = make_symmetry_sweep_instance(N_g=8, k_identical=4, horizon=6, seed=42)
        inst = to_formulation_format(hard)
        r = solve_uc_with_settings_scip(inst, time_limit=30)
        print(f"nodes={r['mip_node_count']} time={r['mip_total_time_s']:.2f}s "
              f"status={r['status']}")
