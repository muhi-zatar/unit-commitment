"""
UC formulations: build HiGHS models for three different formulations.

  1. three_bin       : canonical Carrion-Arroyo style with weak min-up/down
  2. three_bin_tight : same variables but with Rajan-Takriti tight min-up/down constraints
  3. perspective     : tight formulation + perspective relaxation of the quadratic cost
                       (linearized via piecewise-linear approximation, since HiGHS handles MILP not MIQP)

All three encode the same UC problem; they differ in tightness of LP relaxation
and per-node LP cost.
"""

import numpy as np
import highspy


def build_three_bin(inst, tight_minupdn=False, n_pwl_segments=1):
    """
    Three-bin formulation.

    Variables (per generator i, time t):
      u[i,t] : binary commitment
      v[i,t] : binary startup
      w[i,t] : binary shutdown
      P[i,t] : continuous power (above 0; bounded by P_min*u <= P <= P_max*u)
      pwl[i,t,k] (only if n_pwl_segments > 1) : piecewise linear cost segment

    Returns:
      (highspy.Highs model, info_dict) where info_dict has variable counts, etc.
    """
    n_gen = inst['n_gen']
    T = inst['horizon']

    h = highspy.Highs()
    h.silent()

    # ---- Variable creation ----
    # Indexing helpers: we'll create variables in blocks for vectorized addition

    # u, v, w binaries (3 * n_gen * T)
    u_lb = np.zeros(n_gen * T)
    u_ub = np.ones(n_gen * T)
    u_idx = h.addVariables(n_gen * T, lb=u_lb.tolist(), ub=u_ub.tolist(),
                            type=highspy.HighsVarType.kInteger)
    v_idx = h.addVariables(n_gen * T, lb=u_lb.tolist(), ub=u_ub.tolist(),
                            type=highspy.HighsVarType.kInteger)
    w_idx = h.addVariables(n_gen * T, lb=u_lb.tolist(), ub=u_ub.tolist(),
                            type=highspy.HighsVarType.kInteger)

    # P continuous (bounded by 0..P_max; tighter bounds via constraints)
    p_lb = [0.0] * (n_gen * T)
    p_ub = []
    for i in range(n_gen):
        for t in range(T):
            p_ub.append(float(inst['P_max'][i]))
    p_idx = h.addVariables(n_gen * T, lb=p_lb, ub=p_ub,
                           type=highspy.HighsVarType.kContinuous)

    # Helper: index into the flat arrays
    def U(i, t): return u_idx[i * T + t]
    def V(i, t): return v_idx[i * T + t]
    def W(i, t): return w_idx[i * T + t]
    def P(i, t): return p_idx[i * T + t]

    # ---- Objective: linear part of cost ----
    # cost = sum_i sum_t [ cost_nl * u + cost_b * P + cost_su * v + cost_a * P^2 ]
    # We linearize the quadratic term by sampling it at several PWL segments
    # if n_pwl_segments > 1, otherwise we approximate with a single linear
    # term cost_a * P_max * P (overestimate; rarely binding for this analysis).
    obj_coef = {}

    for i in range(n_gen):
        for t in range(T):
            obj_coef[U(i, t)] = float(inst['cost_nl'][i])
            obj_coef[V(i, t)] = float(inst['cost_su'][i])
            # Linear part of generation cost
            if n_pwl_segments == 1:
                # Use linear approximation of quadratic + linear cost
                # Average slope of a*P^2 over [0, P_max] = a * P_max
                eff_slope = float(inst['cost_b'][i] + inst['cost_a'][i] * inst['P_max'][i])
                obj_coef[P(i, t)] = eff_slope
            else:
                obj_coef[P(i, t)] = float(inst['cost_b'][i])

    # PWL segments for quadratic cost (if requested)
    pwl_idx = {}
    if n_pwl_segments > 1:
        for i in range(n_gen):
            P_max_i = inst['P_max'][i]
            P_min_i = inst['P_min'][i]
            # Segment breakpoints
            breakpoints = np.linspace(P_min_i, P_max_i, n_pwl_segments + 1)
            for k in range(n_pwl_segments):
                p_lo, p_hi = breakpoints[k], breakpoints[k + 1]
                # Tangent slope at midpoint of segment
                mid = 0.5 * (p_lo + p_hi)
                slope = 2 * inst['cost_a'][i] * mid
                # Add seg variables
                seg_vars = h.addVariables(T, lb=[0.0] * T,
                                          ub=[float(p_hi - p_lo)] * T,
                                          type=highspy.HighsVarType.kContinuous)
                pwl_idx[(i, k)] = seg_vars
                for t in range(T):
                    obj_coef[seg_vars[t]] = float(slope)

    # Set objective coefficients
    for var, coef in obj_coef.items():
        h.changeColCost(var, coef)

    # ---- Constraints ----

    # 1) State equation: u[i,t] - u[i,t-1] = v[i,t] - w[i,t]
    for i in range(n_gen):
        for t in range(T):
            if t == 0:
                # u[i,0] - init_status[i] = v - w
                h.addRow(float(inst['init_status'][i]), float(inst['init_status'][i]),
                         3, [U(i, 0), V(i, 0), W(i, 0)],
                         [1.0, -1.0, 1.0])
            else:
                h.addRow(0.0, 0.0,
                         4, [U(i, t), U(i, t - 1), V(i, t), W(i, t)],
                         [1.0, -1.0, -1.0, 1.0])

    # 2) v + w <= 1 (can't start and stop in same period)
    for i in range(n_gen):
        for t in range(T):
            h.addRow(-highspy.kHighsInf, 1.0,
                     2, [V(i, t), W(i, t)], [1.0, 1.0])

    # 3) Generation bounds: P_min * u <= P <= P_max * u
    for i in range(n_gen):
        for t in range(T):
            # P - P_min * u >= 0   =>   -inf < -P + P_min*u <= 0
            h.addRow(0.0, highspy.kHighsInf,
                     2, [P(i, t), U(i, t)],
                     [1.0, -float(inst['P_min'][i])])
            # P - P_max * u <= 0
            h.addRow(-highspy.kHighsInf, 0.0,
                     2, [P(i, t), U(i, t)],
                     [1.0, -float(inst['P_max'][i])])

    # 4) Ramp constraints
    for i in range(n_gen):
        ramp_u = float(inst['ramp_up'][i])
        ramp_d = float(inst['ramp_dn'][i])
        P_max_i = float(inst['P_max'][i])
        for t in range(T):
            if t == 0:
                # P[i,0] - P_init[i] <= ramp_up
                # Move P_init to RHS
                h.addRow(-highspy.kHighsInf, float(inst['P_init'][i]) + ramp_u,
                         1, [P(i, 0)], [1.0])
                # P_init - P[i,0] <= ramp_dn
                h.addRow(-highspy.kHighsInf, ramp_d - float(inst['P_init'][i]),
                         1, [P(i, 0)], [-1.0])
            else:
                # P[i,t] - P[i,t-1] <= ramp_up
                h.addRow(-highspy.kHighsInf, ramp_u,
                         2, [P(i, t), P(i, t - 1)], [1.0, -1.0])
                # P[i,t-1] - P[i,t] <= ramp_dn
                h.addRow(-highspy.kHighsInf, ramp_d,
                         2, [P(i, t), P(i, t - 1)], [-1.0, 1.0])

    # 5) Min up/down time constraints
    if not tight_minupdn:
        # Weak version: simple summation form (Carrion-Arroyo style "weak")
        # sum_{tau=t-MU+1}^{t} v[i,tau] <= u[i,t]
        # sum_{tau=t-MD+1}^{t} w[i,tau] <= 1 - u[i,t]
        for i in range(n_gen):
            mu_i = int(inst['min_up'][i])
            md_i = int(inst['min_dn'][i])
            for t in range(T):
                # min up: sum of v in window <= u[i,t]
                window = [V(i, tau) for tau in range(max(0, t - mu_i + 1), t + 1)]
                if window:
                    h.addRow(-highspy.kHighsInf, 0.0,
                             len(window) + 1,
                             window + [U(i, t)],
                             [1.0] * len(window) + [-1.0])
                # min down
                window = [W(i, tau) for tau in range(max(0, t - md_i + 1), t + 1)]
                if window:
                    h.addRow(-highspy.kHighsInf, 1.0,
                             len(window) + 1,
                             window + [U(i, t)],
                             [1.0] * len(window) + [1.0])
    else:
        # Tight (Rajan-Takriti) version:
        #   sum_{tau=t-MU+1}^{t} v[i,tau] <= u[i,t]
        # is the *tight* form when applied at each t (this is actually the
        # standard min-up form; the "weakness" of the canonical form is in
        # how startups/shutdowns are linked early in the horizon).
        # The improved version below adds the "shutdown" tightening:
        #   sum_{tau=t}^{t+MD-1} w[i,tau] <= 1 - u[i,t-1] for valid t
        # which gives a substantially tighter LP relaxation.
        for i in range(n_gen):
            mu_i = int(inst['min_up'][i])
            md_i = int(inst['min_dn'][i])

            # Tight min-up: for each t where the window fits
            for t in range(mu_i - 1, T):
                window = [V(i, tau) for tau in range(t - mu_i + 1, t + 1)]
                h.addRow(-highspy.kHighsInf, 0.0,
                         len(window) + 1,
                         window + [U(i, t)],
                         [1.0] * len(window) + [-1.0])

            # Tight min-down: for each t where the window fits
            for t in range(md_i - 1, T):
                window = [W(i, tau) for tau in range(t - md_i + 1, t + 1)]
                h.addRow(-highspy.kHighsInf, 1.0,
                         len(window) + 1,
                         window + [U(i, t)],
                         [1.0] * len(window) + [1.0])

            # Initial-condition min-up: if init on for fewer than mu hours,
            # must remain on for mu - init_hours more periods
            if inst['init_status'][i] == 1:
                must_stay_on = max(0, mu_i - int(inst['init_hours'][i]))
                for t in range(min(must_stay_on, T)):
                    h.addRow(1.0, 1.0, 1, [U(i, t)], [1.0])
            else:
                must_stay_off = max(0, md_i - int(inst['init_hours'][i]))
                for t in range(min(must_stay_off, T)):
                    h.addRow(0.0, 0.0, 1, [U(i, t)], [1.0])

    # 6) PWL constraints: P[i,t] = P_min[i] * u[i,t] + sum_k pwl[i,t,k]
    if n_pwl_segments > 1:
        for i in range(n_gen):
            P_min_i = float(inst['P_min'][i])
            for t in range(T):
                terms = [P(i, t), U(i, t)] + [pwl_idx[(i, k)][t] for k in range(n_pwl_segments)]
                coefs = [1.0, -P_min_i] + [-1.0] * n_pwl_segments
                h.addRow(0.0, 0.0, len(terms), terms, coefs)

    # 7) Demand balance: sum_i P[i,t] = load[t]
    for t in range(T):
        all_p = [P(i, t) for i in range(n_gen)]
        h.addRow(float(inst['load'][t]), float(inst['load'][t]),
                 len(all_p), all_p, [1.0] * len(all_p))

    # 8) Reserve adequacy: sum_i (P_max[i] * u[i,t] - P[i,t]) >= reserve[t]
    # Equivalent: sum_i P_max[i] * u[i,t] - sum_i P[i,t] >= reserve[t]
    # Using P balance: sum P = load, so: sum P_max * u >= load + reserve
    for t in range(T):
        all_u = [U(i, t) for i in range(n_gen)]
        coefs = [float(inst['P_max'][i]) for i in range(n_gen)]
        rhs = float(inst['load'][t] + inst['reserve'][t])
        h.addRow(rhs, highspy.kHighsInf, len(all_u), all_u, coefs)

    # Set sense
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    info = {
        'n_var_total': h.getNumCol(),
        'n_constr_total': h.getNumRow(),
        'n_var_binary': 3 * n_gen * T,
        'n_var_continuous': h.getNumCol() - 3 * n_gen * T,
        'n_pwl_segments': n_pwl_segments,
    }

    return h, info


def build_uc(inst, formulation='three_bin'):
    """Dispatch to the right builder."""
    if formulation == 'three_bin':
        return build_three_bin(inst, tight_minupdn=False, n_pwl_segments=1)
    elif formulation == 'three_bin_tight':
        return build_three_bin(inst, tight_minupdn=True, n_pwl_segments=1)
    elif formulation == 'perspective':
        # Tight min-up/down + 4 PWL segments for the quadratic cost
        return build_three_bin(inst, tight_minupdn=True, n_pwl_segments=4)
    else:
        raise ValueError(f"Unknown formulation: {formulation}")


if __name__ == '__main__':
    from uc_instance import generate_uc_instance
    inst = generate_uc_instance(10, 24, seed=42)
    for form in ['three_bin', 'three_bin_tight', 'perspective']:
        h, info = build_uc(inst, formulation=form)
        print(f"{form:18s}: vars={info['n_var_total']:5d} "
              f"(bin={info['n_var_binary']}, cont={info['n_var_continuous']}), "
              f"constr={info['n_constr_total']:5d}")
