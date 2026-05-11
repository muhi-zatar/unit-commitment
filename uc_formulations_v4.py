"""
V4 UC formulation builder: three_bin with four optional fidelity features.

Adapted from uc_formulations.build_three_bin. Adds four binary toggles via
a `features` dict:

    F1 — piecewise quadratic cost (4 segments)        (n_pwl_segments=4)
    F2 — separate startup/shutdown ramp limits        (Knueven §3.3)
    F3 — lag-dependent startup costs (3 categories)   (startup-arc binaries)
    F4 — must-run enforcement                         (u[i,t] == 1 forced)

When all four flags are False, the model is identical to the v3 three_bin
(except for the explicit cost_a, which v3 also linearises at n_pwl_segments=1).

Expected new fields on inst (only when corresponding flag is True):

    F1: 'cost_a'                       (already present in v3 inst dicts)
    F2: 'ramp_startup_limit'           array, length n_gen
        'ramp_shutdown_limit'          array, length n_gen
    F3: 'startup_categories'           list[n_gen] of list[(lag_lo, lag_hi, cost)]
                                       categories must cover [1, T+horizon]
    F4: 'must_run'                     boolean array, length n_gen

This file does NOT modify uc_formulations.py — it is a parallel module.
"""

import numpy as np
import highspy


DEFAULT_FEATURES = {'F1': False, 'F2': False, 'F3': False, 'F4': False}
N_PWL_SEGMENTS_F1 = 4


def build_three_bin_v4(inst, features=None, tight_minupdn=False):
    """
    Build a three-bin UC model with optional fidelity features.

    Returns (highspy.Highs model, info_dict).
    """
    f = dict(DEFAULT_FEATURES)
    if features:
        f.update(features)

    n_gen = inst['n_gen']
    T = inst['horizon']

    h = highspy.Highs()
    h.silent()

    # ---- Binary commitment/startup/shutdown ----
    u_idx = h.addVariables(n_gen * T,
                           lb=[0.0] * (n_gen * T), ub=[1.0] * (n_gen * T),
                           type=highspy.HighsVarType.kInteger)
    v_idx = h.addVariables(n_gen * T,
                           lb=[0.0] * (n_gen * T), ub=[1.0] * (n_gen * T),
                           type=highspy.HighsVarType.kInteger)
    w_idx = h.addVariables(n_gen * T,
                           lb=[0.0] * (n_gen * T), ub=[1.0] * (n_gen * T),
                           type=highspy.HighsVarType.kInteger)

    # ---- Continuous power ----
    p_lb = [0.0] * (n_gen * T)
    p_ub = []
    for i in range(n_gen):
        for _ in range(T):
            p_ub.append(float(inst['P_max'][i]))
    p_idx = h.addVariables(n_gen * T, lb=p_lb, ub=p_ub,
                           type=highspy.HighsVarType.kContinuous)

    def U(i, t): return u_idx[i * T + t]
    def V(i, t): return v_idx[i * T + t]
    def W(i, t): return w_idx[i * T + t]
    def P(i, t): return p_idx[i * T + t]

    obj_coef = {}

    # ---- F1: PWL cost segments (4 if enabled, else 1 linear) ----
    n_pwl = N_PWL_SEGMENTS_F1 if f['F1'] else 1
    pwl_idx = {}

    if n_pwl > 1:
        for i in range(n_gen):
            P_max_i = float(inst['P_max'][i])
            P_min_i = float(inst['P_min'][i])
            breakpoints = np.linspace(P_min_i, P_max_i, n_pwl + 1)
            for k in range(n_pwl):
                p_lo, p_hi = breakpoints[k], breakpoints[k + 1]
                mid = 0.5 * (p_lo + p_hi)
                slope = float(inst['cost_b'][i]) + 2.0 * float(inst['cost_a'][i]) * mid
                seg_vars = h.addVariables(T,
                                          lb=[0.0] * T,
                                          ub=[float(p_hi - p_lo)] * T,
                                          type=highspy.HighsVarType.kContinuous)
                pwl_idx[(i, k)] = seg_vars
                for t in range(T):
                    obj_coef[seg_vars[t]] = slope

    # ---- F3: startup arc variables (if enabled) ----
    delta_idx = {}      # (i, t, s) -> highs var index
    n_startup_cats = 0
    if f['F3']:
        cats = inst['startup_categories']
        # All units must declare the same number of categories
        n_startup_cats = len(cats[0])
        for i in range(n_gen):
            assert len(cats[i]) == n_startup_cats, \
                f"unit {i} has {len(cats[i])} startup categories; expected {n_startup_cats}"
            for s in range(n_startup_cats):
                arc_vars = h.addVariables(T,
                                          lb=[0.0] * T, ub=[1.0] * T,
                                          type=highspy.HighsVarType.kInteger)
                delta_idx[(i, s)] = arc_vars

    # ---- Objective: linear part ----
    for i in range(n_gen):
        for t in range(T):
            obj_coef[U(i, t)] = float(inst['cost_nl'][i])
            # If F3, startup cost is paid via δ instead of v
            if not f['F3']:
                obj_coef[V(i, t)] = float(inst['cost_su'][i])
            if n_pwl == 1:
                # Single-segment linear approximation (matches v3 default)
                eff_slope = float(inst['cost_b'][i]) + \
                            float(inst['cost_a'][i]) * float(inst['P_max'][i])
                obj_coef[P(i, t)] = eff_slope

    if f['F3']:
        cats = inst['startup_categories']
        for i in range(n_gen):
            for s, (lag_lo, lag_hi, cost) in enumerate(cats[i]):
                for t in range(T):
                    obj_coef[delta_idx[(i, s)][t]] = float(cost)

    for var, coef in obj_coef.items():
        h.changeColCost(var, coef)

    # ---- 1) State equation ----
    for i in range(n_gen):
        for t in range(T):
            if t == 0:
                h.addRow(float(inst['init_status'][i]), float(inst['init_status'][i]),
                         3, [U(i, 0), V(i, 0), W(i, 0)],
                         [1.0, -1.0, 1.0])
            else:
                h.addRow(0.0, 0.0,
                         4, [U(i, t), U(i, t - 1), V(i, t), W(i, t)],
                         [1.0, -1.0, -1.0, 1.0])

    # ---- 2) Can't start and stop in same period ----
    for i in range(n_gen):
        for t in range(T):
            h.addRow(-highspy.kHighsInf, 1.0,
                     2, [V(i, t), W(i, t)], [1.0, 1.0])

    # ---- 3) Generation bounds (with F2 startup/shutdown ramp adjustment) ----
    for i in range(n_gen):
        P_max_i = float(inst['P_max'][i])
        P_min_i = float(inst['P_min'][i])
        for t in range(T):
            # P >= P_min * u
            h.addRow(0.0, highspy.kHighsInf,
                     2, [P(i, t), U(i, t)], [1.0, -P_min_i])

            if f['F2']:
                # Knueven §3.3 tight upper bound:
                #   P[i,t] <= P_max*u[i,t]
                #            - (P_max - SU)*v[i,t]
                #            - (P_max - SD)*w[i,t+1]
                SU = float(inst['ramp_startup_limit'][i])
                SD = float(inst['ramp_shutdown_limit'][i])
                terms = [P(i, t), U(i, t), V(i, t)]
                coefs = [1.0, -P_max_i, (P_max_i - SU)]
                if t + 1 < T:
                    terms.append(W(i, t + 1))
                    coefs.append((P_max_i - SD))
                h.addRow(-highspy.kHighsInf, 0.0, len(terms), terms, coefs)
            else:
                # v3-style: P <= P_max * u
                h.addRow(-highspy.kHighsInf, 0.0,
                         2, [P(i, t), U(i, t)], [1.0, -P_max_i])

    # ---- 4) Ramp constraints ----
    for i in range(n_gen):
        ramp_u = float(inst['ramp_up'][i])
        ramp_d = float(inst['ramp_dn'][i])
        for t in range(T):
            if t == 0:
                pi0 = float(inst['P_init'][i])
                h.addRow(-highspy.kHighsInf, pi0 + ramp_u,
                         1, [P(i, 0)], [1.0])
                h.addRow(-highspy.kHighsInf, ramp_d - pi0,
                         1, [P(i, 0)], [-1.0])
            else:
                h.addRow(-highspy.kHighsInf, ramp_u,
                         2, [P(i, t), P(i, t - 1)], [1.0, -1.0])
                h.addRow(-highspy.kHighsInf, ramp_d,
                         2, [P(i, t), P(i, t - 1)], [-1.0, 1.0])

    # ---- 5) Min up / down (weak form, same as v3 three_bin default) ----
    if not tight_minupdn:
        for i in range(n_gen):
            mu_i = int(inst['min_up'][i])
            md_i = int(inst['min_dn'][i])
            for t in range(T):
                window = [V(i, tau) for tau in range(max(0, t - mu_i + 1), t + 1)]
                if window:
                    h.addRow(-highspy.kHighsInf, 0.0,
                             len(window) + 1,
                             window + [U(i, t)],
                             [1.0] * len(window) + [-1.0])
                window = [W(i, tau) for tau in range(max(0, t - md_i + 1), t + 1)]
                if window:
                    h.addRow(-highspy.kHighsInf, 1.0,
                             len(window) + 1,
                             window + [U(i, t)],
                             [1.0] * len(window) + [1.0])
    else:
        for i in range(n_gen):
            mu_i = int(inst['min_up'][i])
            md_i = int(inst['min_dn'][i])
            for t in range(mu_i - 1, T):
                window = [V(i, tau) for tau in range(t - mu_i + 1, t + 1)]
                h.addRow(-highspy.kHighsInf, 0.0,
                         len(window) + 1, window + [U(i, t)],
                         [1.0] * len(window) + [-1.0])
            for t in range(md_i - 1, T):
                window = [W(i, tau) for tau in range(t - md_i + 1, t + 1)]
                h.addRow(-highspy.kHighsInf, 1.0,
                         len(window) + 1, window + [U(i, t)],
                         [1.0] * len(window) + [1.0])

    # ---- 6) PWL cost: P = P_min*u + sum_k seg ----
    if n_pwl > 1:
        for i in range(n_gen):
            P_min_i = float(inst['P_min'][i])
            for t in range(T):
                terms = [P(i, t), U(i, t)] + [pwl_idx[(i, k)][t] for k in range(n_pwl)]
                coefs = [1.0, -P_min_i] + [-1.0] * n_pwl
                h.addRow(0.0, 0.0, len(terms), terms, coefs)

    # ---- 7) Demand balance ----
    for t in range(T):
        all_p = [P(i, t) for i in range(n_gen)]
        h.addRow(float(inst['load'][t]), float(inst['load'][t]),
                 len(all_p), all_p, [1.0] * len(all_p))

    # ---- 8) Reserve adequacy ----
    for t in range(T):
        all_u = [U(i, t) for i in range(n_gen)]
        coefs = [float(inst['P_max'][i]) for i in range(n_gen)]
        rhs = float(inst['load'][t] + inst['reserve'][t])
        h.addRow(rhs, highspy.kHighsInf, len(all_u), all_u, coefs)

    # ---- F3: startup arc linking ----
    # sum_s delta[i,t,s] = v[i,t]  for all (i,t)
    # For each category s with bracket [lag_lo, lag_hi]:
    #   delta[i,t,s] <= sum_{tau in window} w[i,tau]   (must have shut down in window)
    # The widest category is the residual (no window constraint).
    if f['F3']:
        cats = inst['startup_categories']
        for i in range(n_gen):
            for t in range(T):
                # delta sums to v
                terms = [delta_idx[(i, s)][t] for s in range(n_startup_cats)] + [V(i, t)]
                coefs = [1.0] * n_startup_cats + [-1.0]
                h.addRow(0.0, 0.0, len(terms), terms, coefs)

                # Bracket constraints: for the hot/warm cats, constrain by w window
                for s in range(n_startup_cats - 1):  # last cat is residual
                    lag_lo, lag_hi, _ = cats[i][s]
                    win = [W(i, tau) for tau in
                           range(max(0, t - int(lag_hi) + 1), max(0, t - int(lag_lo) + 1))]
                    if win:
                        terms = [delta_idx[(i, s)][t]] + win
                        coefs = [1.0] + [-1.0] * len(win)
                        h.addRow(-highspy.kHighsInf, 0.0,
                                 len(terms), terms, coefs)
                    else:
                        # No valid lag window → category infeasible at this t
                        h.addRow(0.0, 0.0, 1, [delta_idx[(i, s)][t]], [1.0])

    # ---- F4: must-run ----
    if f['F4']:
        must = inst['must_run']
        for i in range(n_gen):
            if not must[i]:
                continue
            for t in range(T):
                h.addRow(1.0, 1.0, 1, [U(i, t)], [1.0])

    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    info = {
        'n_var_total': h.getNumCol(),
        'n_constr_total': h.getNumRow(),
        'n_var_binary': 3 * n_gen * T + (n_gen * T * n_startup_cats if f['F3'] else 0),
        'n_var_continuous': h.getNumCol() - 3 * n_gen * T -
                            (n_gen * T * n_startup_cats if f['F3'] else 0),
        'n_pwl_segments': n_pwl,
        'features': dict(f),
    }
    return h, info


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uc_hard_instance import generate_hard_uc, FLEET_RECIPES
    from instance_adapter import to_formulation_format
    import numpy as np

    hard = generate_hard_uc(FLEET_RECIPES['diverse_20'],
                            horizon=12, reserve_frac=0.15,
                            plateau_factor=1.0, seed=42)
    inst = to_formulation_format(hard)
    # Stub the v4-specific fields so all flags can be tested
    inst['ramp_startup_limit'] = inst['P_min'] + 0.3 * (inst['P_max'] - inst['P_min'])
    inst['ramp_shutdown_limit'] = inst['P_min'] + 0.3 * (inst['P_max'] - inst['P_min'])
    inst['must_run'] = np.zeros(inst['n_gen'], dtype=bool)
    inst['must_run'][:3] = True  # first three are must-run
    # 3 categories: hot (1-3h), warm (4-8h), cold (9+h, residual — set lag_hi very high)
    inst['startup_categories'] = [
        [(1, 3, float(inst['cost_su'][i])),
         (4, 8, float(inst['cost_su'][i]) * 1.5),
         (9, 9999, float(inst['cost_su'][i]) * 2.5)]
        for i in range(inst['n_gen'])
    ]

    from itertools import product
    print(f"{'features':10s} {'n_var':>6s} {'n_constr':>8s} {'n_bin':>6s}")
    print('-' * 40)
    for combo in product([0, 1], repeat=4):
        feats = dict(zip(['F1', 'F2', 'F3', 'F4'], [bool(x) for x in combo]))
        h, info = build_three_bin_v4(inst, features=feats)
        label = ''.join(str(c) for c in combo)
        print(f"{label:10s} {info['n_var_total']:6d} "
              f"{info['n_constr_total']:8d} {info['n_var_binary']:6d}")
