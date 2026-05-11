"""
Adapter: convert hard-instance format -> formulations.py format.

The original uc_formulations.py expects field names like P_max, cost_a,
init_status, P_init, etc. Our hard generator uses pmax, a_cost, initial_on,
etc. This module converts between them.
"""

import numpy as np


def to_formulation_format(hard_inst):
    """Convert a hard_inst dict to the format uc_formulations.py expects."""
    n_gen = hard_inst['n_gen']
    pmax = hard_inst['pmax']
    pmin = hard_inst['pmin']
    a = hard_inst['a_cost']
    b = hard_inst['b_cost']
    c = hard_inst['c_cost']
    su = hard_inst['su_cost']

    # Initial power: midpoint of [pmin, pmax] for committed units, 0 for off
    P_init = np.where(hard_inst['initial_on'] == 1,
                      0.5 * (pmin + pmax), 0.0)

    # Ramp up/down (we use the same 'ramp' for both directions)
    ramp = hard_inst['ramp']

    return {
        'n_gen': n_gen,
        'horizon': hard_inst['horizon'],
        'demand': hard_inst['demand'],
        'load': hard_inst['demand'],     # alias used in some formulation rows
        'reserve': hard_inst['reserve'],
        'P_min': pmin,
        'P_max': pmax,
        'cost_nl': c,        # no-load cost
        'cost_su': su,       # startup cost
        'cost_b': b,         # linear coefficient
        'cost_a': a,         # quadratic coefficient
        'min_up': hard_inst['min_up'],
        'min_dn': hard_inst['min_dn'],
        'ramp_up': ramp,
        'ramp_dn': ramp,
        'init_status': hard_inst['initial_on'],
        'init_hours': np.where(hard_inst['initial_on'] == 1,
                                hard_inst['initial_uptime'],
                                hard_inst['initial_dntime']),
        'P_init': P_init,
        # Pass through metadata for analysis
        'fleet_spec': hard_inst.get('fleet_spec', None),
        'symmetry_score': hard_inst.get('symmetry_score', 0),
        'plateau_factor': hard_inst.get('plateau_factor', 1.0),
        'reserve_frac': hard_inst.get('reserve_frac', 0.10),
    }


if __name__ == '__main__':
    from uc_hard_instance import generate_hard_uc, FLEET_RECIPES
    from uc_solver import solve_uc

    print("=== Solver smoke test on hard instances ===\n")
    for recipe_name in ['diverse_20', 'sym8_24', 'sym_extreme_20']:
        print(f"\n>>> Recipe: {recipe_name}")
        recipe = FLEET_RECIPES[recipe_name]
        hard = generate_hard_uc(recipe, horizon=24, reserve_frac=0.20,
                                plateau_factor=0.1, seed=42)
        inst = to_formulation_format(hard)
        print(f"    n_gen={inst['n_gen']}, sym_score={hard['symmetry_score']}")
        for form in ['three_bin', 'three_bin_tight']:
            try:
                r = solve_uc(inst, formulation=form, time_limit=60, gap_tol=0.001)
                print(f"    {form:18s}: time={r['mip_total_time_s']:6.2f}s, "
                      f"nodes={r['mip_node_count']:6d}, "
                      f"gap={r['root_lp_gap_pct']:.3f}%, "
                      f"status={r['status']}")
            except Exception as e:
                print(f"    {form:18s}: ERROR: {e}")
