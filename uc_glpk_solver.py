"""
UC solver using GLPK via PuLP.

GLPK is far less aggressive than HiGHS:
  - Simpler presolve
  - Almost no primal heuristics by default
  - Basic Gomory + MIR cuts only when explicitly enabled
  - No fancy symmetry detection

This means symmetric instances will actually branch, exposing the empirical
N_nodes scaling that the master cost expression predicts.

The solver writes the model to an MPS file and calls glpsol.
We capture node counts and timing from glpsol's output.
"""

import time
import re
import subprocess
import tempfile
import os
import numpy as np
import pulp


def build_uc_pulp(inst, formulation='three_bin'):
    """
    Build the UC LP/MILP model in PuLP.
    Returns the LpProblem object.

    Formulations:
      'three_bin'       — loose min up/down (sequence of pairwise indicators)
      'three_bin_tight' — Rajan-Takriti tight min up/down (cumulative sum form)

    Quadratic costs are linearized with 5-segment PWL secant approximation.
    """
    n_gen = inst['n_gen']
    T = inst['horizon']
    pmax = inst['P_max']
    pmin = inst['P_min']
    a = inst['cost_a']
    b = inst['cost_b']
    c_nl = inst['cost_nl']
    c_su = inst['cost_su']
    min_up = inst['min_up']
    min_dn = inst['min_dn']
    ramp_up = inst['ramp_up']
    ramp_dn = inst['ramp_dn']
    demand = inst['load']
    reserve = inst['reserve']
    init_status = inst['init_status']

    n_seg = 5
    prob = pulp.LpProblem("UC", pulp.LpMinimize)

    # Variables
    u = pulp.LpVariable.dicts("u", ((i, t) for i in range(n_gen) for t in range(T)),
                               cat='Binary')
    v = pulp.LpVariable.dicts("v", ((i, t) for i in range(n_gen) for t in range(T)),
                               cat='Binary')
    w = pulp.LpVariable.dicts("w", ((i, t) for i in range(n_gen) for t in range(T)),
                               cat='Binary')
    p = pulp.LpVariable.dicts("p", ((i, t) for i in range(n_gen) for t in range(T)),
                               lowBound=0, upBound=None)
    q = pulp.LpVariable.dicts("q", ((i, t) for i in range(n_gen) for t in range(T)),
                               lowBound=0)

    # Objective: no-load + startup + linear power + quadratic-segment cost
    prob += pulp.lpSum(c_nl[i] * u[(i, t)] + c_su[i] * v[(i, t)] +
                       b[i] * p[(i, t)] + q[(i, t)]
                       for i in range(n_gen) for t in range(T))

    # 1) Demand balance
    for t in range(T):
        prob += pulp.lpSum(p[(i, t)] for i in range(n_gen)) == demand[t]

    # 2) Reserve adequacy: sum (pmax_i * u_it - p_it) >= reserve_t
    for t in range(T):
        prob += pulp.lpSum(pmax[i] * u[(i, t)] - p[(i, t)]
                          for i in range(n_gen)) >= reserve[t]

    # 3) Capacity bounds: pmin*u <= p <= pmax*u
    for i in range(n_gen):
        for t in range(T):
            prob += p[(i, t)] >= pmin[i] * u[(i, t)]
            prob += p[(i, t)] <= pmax[i] * u[(i, t)]

    # 4) State equation
    for i in range(n_gen):
        for t in range(T):
            if t == 0:
                prob += u[(i, t)] - int(init_status[i]) == v[(i, t)] - w[(i, t)]
            else:
                prob += u[(i, t)] - u[(i, t - 1)] == v[(i, t)] - w[(i, t)]
            prob += v[(i, t)] + w[(i, t)] <= 1

    # 5) Min up/down
    if formulation == 'three_bin':
        # Loose: each startup blocks shutdowns for the next min_up periods
        for i in range(n_gen):
            for t in range(T):
                for k in range(1, int(min_up[i])):
                    if t + k < T:
                        prob += v[(i, t)] - u[(i, t + k)] <= 0
                for k in range(1, int(min_dn[i])):
                    if t + k < T:
                        prob += w[(i, t)] + u[(i, t + k)] <= 1
    elif formulation == 'three_bin_tight':
        for i in range(n_gen):
            for t in range(T):
                start = max(0, t - int(min_up[i]) + 1)
                prob += pulp.lpSum(v[(i, tau)] for tau in range(start, t + 1)) - u[(i, t)] <= 0
                start = max(0, t - int(min_dn[i]) + 1)
                prob += pulp.lpSum(w[(i, tau)] for tau in range(start, t + 1)) + u[(i, t)] <= 1
    else:
        raise ValueError(f"Unknown formulation: {formulation}")

    # 6) Ramp limits (between consecutive periods)
    for i in range(n_gen):
        for t in range(1, T):
            prob += p[(i, t)] - p[(i, t - 1)] <= ramp_up[i]
            prob += p[(i, t - 1)] - p[(i, t)] <= ramp_dn[i]

    # 7) PWL quadratic cost: q_it >= secant_k * p_it + intercept_k * u_it
    for i in range(n_gen):
        bps = np.linspace(pmin[i], pmax[i], n_seg + 1)
        for k in range(n_seg):
            p1, p2 = bps[k], bps[k + 1]
            slope = a[i] * (p1 + p2)
            intercept = -a[i] * p1 * p2
            for t in range(T):
                prob += q[(i, t)] - slope * p[(i, t)] - intercept * u[(i, t)] >= 0

    return prob


def parse_glpk_log(log_text):
    """Extract node count, time, status, optimal value from glpsol output."""
    info = {
        'n_nodes': None,
        'n_simplex_iters': None,
        'final_obj': None,
        'final_lower_bound': None,
        'status': 'unknown',
    }

    # GLPK reports nodes in lines like:
    #   "+   1411: mip =   1.512517224e+05 >=   1.511076378e+05 < 0.1% (3; 44)"
    #   "+   729: >>>>>   1.512517224e+05 >=   1.506914205e+05   0.4% (14; 0)"
    # Pull all such lines and take the last for final values.
    pattern = re.compile(
        r'\+\s*(\d+):\s*(?:mip|>>>>>)\s*=?\s*([\d.eE+-]+)\s*>=\s*([\d.eE+-]+)'
    )
    matches = list(pattern.finditer(log_text))
    if matches:
        last = matches[-1]
        info['n_nodes'] = int(last.group(1))
        try:
            info['final_obj'] = float(last.group(2))
        except ValueError:
            pass
        try:
            info['final_lower_bound'] = float(last.group(3))
        except ValueError:
            pass

    # Total simplex iterations across LP relaxation and node LPs are not
    # reliably reported in MIP runs; leave as the explicit field if present
    simp = re.search(r'^\s*(\d+):\s*obj\s*=', log_text, re.MULTILINE)
    if simp:
        info['n_simplex_iters'] = int(simp.group(1))

    if 'OPTIMAL SOLUTION' in log_text or 'INTEGER OPTIMAL' in log_text \
            or 'RELATIVE MIP GAP TOLERANCE REACHED' in log_text:
        info['status'] = 'optimal'
    elif 'TIME LIMIT' in log_text:
        info['status'] = 'time_limit'
    elif 'NO INTEGER' in log_text or 'INFEASIBLE' in log_text:
        info['status'] = 'infeasible'

    return info


def solve_uc_glpk(inst, formulation='three_bin', time_limit=120,
                   mip_gap=0.001, presolve=False, cuts=False,
                   verbose=False):
    """
    Solve UC instance using GLPK via direct LP file interface.

    Settings:
      presolve : whether to enable GLPK's presolver
      cuts     : whether to enable Gomory + MIR cuts
    """
    # ---- Build PuLP model ----
    t0 = time.perf_counter()
    prob = build_uc_pulp(inst, formulation=formulation)
    build_time = time.perf_counter() - t0

    n_var_total = len(prob.variables())
    n_constr = len(prob.constraints)
    n_var_binary = sum(1 for v in prob.variables() if v.cat == 'Binary')

    # ---- Solve LP relaxation first for the gap ----
    t0 = time.perf_counter()
    prob_lp = prob.copy()
    for var in prob_lp.variables():
        if var.cat == 'Binary':
            var.cat = 'Continuous'
            var.lowBound = 0
            var.upBound = 1
    glpk_lp = pulp.GLPK_CMD(msg=False, options=['--lp'])
    prob_lp.solve(glpk_lp)
    lp_relax_time = time.perf_counter() - t0
    lp_relax_obj = pulp.value(prob_lp.objective)

    # ---- Solve MIP via GLPK ----
    options = ['--mipgap', str(mip_gap), '--tmlim', str(int(time_limit))]
    if presolve:
        options.append('--presol')
    else:
        options.append('--nopresol')
    if cuts:
        options.extend(['--cuts'])
    else:
        # GLPK has cuts off by default
        pass

    glpk = pulp.GLPK_CMD(msg=False, options=options, keepFiles=False)

    # PuLP doesn't expose glpsol's log directly; capture by writing files manually
    with tempfile.TemporaryDirectory() as td:
        lp_path = os.path.join(td, 'model.lp')
        sol_path = os.path.join(td, 'model.sol')
        prob.writeLP(lp_path)

        cmd = ['glpsol', '--lp', lp_path, '-o', sol_path] + options
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=time_limit + 30)
        mip_time = time.perf_counter() - t0
        log_text = result.stdout + result.stderr

    info = parse_glpk_log(log_text)
    mip_obj = info['final_obj']
    nodes = info['n_nodes'] if info['n_nodes'] is not None else 0
    simp_iters = info['n_simplex_iters'] if info['n_simplex_iters'] is not None else 0
    status = info['status']
    converged = status == 'optimal'

    if mip_obj is None or lp_relax_obj is None:
        root_lp_gap_pct = float('nan')
    elif mip_obj > 0:
        root_lp_gap_pct = (mip_obj - lp_relax_obj) / mip_obj * 100
    else:
        root_lp_gap_pct = 0.0

    return {
        'formulation': formulation,
        'solver': 'GLPK',
        'n_gen': inst['n_gen'],
        'horizon': inst['horizon'],
        'n_var': n_var_total,
        'n_constr': n_constr,
        'n_var_binary': n_var_binary,
        'build_time_s': build_time,
        'lp_relax_time_s': lp_relax_time,
        'lp_relax_obj': lp_relax_obj,
        'mip_total_time_s': mip_time,
        'mip_obj': mip_obj,
        'mip_node_count': nodes,
        'mip_simplex_iters': simp_iters,
        'converged': converged,
        'status': status,
        'root_lp_gap_pct': root_lp_gap_pct,
        'presolve': presolve,
        'cuts': cuts,
    }


if __name__ == '__main__':
    from uc_hard_instance import generate_hard_uc, FLEET_RECIPES
    from instance_adapter import to_formulation_format

    print("=== GLPK solver smoke test ===\n")
    print(f"{'instance':22s} | {'form':18s} | {'time':>8s} | {'nodes':>7s} | {'gap':>8s} | status")
    print('-' * 88)

    for recipe_name in ['diverse_20', 'sym8_24']:
        recipe = FLEET_RECIPES[recipe_name]
        for plateau in [1.0, 0.0]:
            hard = generate_hard_uc(recipe, horizon=24, reserve_frac=0.15,
                                    plateau_factor=plateau, seed=42)
            inst = to_formulation_format(hard)
            for form in ['three_bin', 'three_bin_tight']:
                r = solve_uc_glpk(inst, formulation=form, time_limit=60)
                tag = f"{recipe_name} p={plateau}"
                print(f"{tag:22s} | {form:18s} | "
                      f"{r['mip_total_time_s']:7.2f}s | "
                      f"{r['mip_node_count']:7d} | "
                      f"{r['root_lp_gap_pct']:7.3f}% | "
                      f"{r['status']}")
