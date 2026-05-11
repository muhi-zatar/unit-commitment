"""
HARD UC instance generator with controllable difficulty levers.

Three knobs control how many B&C nodes the solver explores:

  symmetry_groups : how many groups of identical generators to create
                    Higher = more symmetric LP solutions = more branching
                    e.g. (8, 4) means 8 identical units of one type, 4 of another

  reserve_frac    : fraction of demand required as spinning reserve
                    Higher = tighter coupling between commitment decisions
                    0.10 is benign; 0.20-0.30 forces binding constraints

  cost_plateau    : 0.0 to 1.0 — how compressed the cost curves are
                    1.0 = baseline diversity (easy)
                    0.1 = nearly-identical marginal costs (hard, many local optima)

The original generator used n_gen with random parameters per generator; this one
takes a list of (count, type) tuples to control symmetry directly.
"""

import numpy as np


# Predefined "type templates" for generator classes
TYPE_TEMPLATES = {
    'baseload_large': {
        'pmax_range': (400, 600), 'pmin_frac': (0.40, 0.50),
        'a_range': (0.0001, 0.0002), 'b_range': (18, 22),
        'c_range': (250, 350), 'su_range': (8000, 12000),
        'min_up_range': (8, 12), 'min_dn_range': (8, 12),
        'ramp_frac': 0.30,
    },
    'baseload_med': {
        'pmax_range': (200, 350), 'pmin_frac': (0.35, 0.45),
        'a_range': (0.0002, 0.0004), 'b_range': (22, 28),
        'c_range': (150, 250), 'su_range': (3000, 6000),
        'min_up_range': (5, 8), 'min_dn_range': (5, 8),
        'ramp_frac': 0.40,
    },
    'midmerit': {
        'pmax_range': (100, 200), 'pmin_frac': (0.30, 0.40),
        'a_range': (0.0004, 0.0008), 'b_range': (28, 38),
        'c_range': (80, 180), 'su_range': (1500, 3000),
        'min_up_range': (3, 6), 'min_dn_range': (3, 6),
        'ramp_frac': 0.50,
    },
    'peaker': {
        'pmax_range': (50, 100), 'pmin_frac': (0.15, 0.25),
        'a_range': (0.0008, 0.0015), 'b_range': (50, 75),
        'c_range': (40, 100), 'su_range': (300, 800),
        'min_up_range': (1, 3), 'min_dn_range': (1, 3),
        'ramp_frac': 0.80,
    },
}


def _sample_unit(template, rng, plateau_factor=1.0):
    """
    Sample one generator's parameters from the template.

    plateau_factor in (0, 1]: scales the random spread within ranges.
    1.0 = full randomness, 0.0 = identical units within the template.
    """
    def lerp_range(rng_a, rng_b):
        midpoint = 0.5 * (rng_a + rng_b)
        half = 0.5 * (rng_b - rng_a) * plateau_factor
        return rng.uniform(midpoint - half, midpoint + half)

    pmax = lerp_range(*template['pmax_range'])
    pmin_frac = lerp_range(*template['pmin_frac'])
    return {
        'pmax': pmax,
        'pmin': pmax * pmin_frac,
        'a_cost': lerp_range(*template['a_range']),
        'b_cost': lerp_range(*template['b_range']),
        'c_cost': lerp_range(*template['c_range']),
        'su_cost': lerp_range(*template['su_range']),
        'sd_cost': lerp_range(*template['su_range']) * 0.3,
        'min_up': int(round(lerp_range(*template['min_up_range']))),
        'min_dn': int(round(lerp_range(*template['min_dn_range']))),
        'ramp': pmax * template['ramp_frac'],
    }


def generate_hard_uc(fleet_spec, horizon=24, reserve_frac=0.15,
                     plateau_factor=1.0, seed=0):
    """
    Generate a UC instance with controlled symmetry.

    Parameters
    ----------
    fleet_spec : list of (count, type_name)
        e.g. [(8, 'baseload_large'), (12, 'midmerit'), (6, 'peaker')]
        Each (count, type) means `count` IDENTICAL generators of that type
        (subject to plateau_factor smoothing).
    horizon : T in hours
    reserve_frac : fraction of demand as spinning reserve
    plateau_factor : 0.0 to 1.0; lower = more symmetry within each type group
    seed : RNG seed
    """
    rng = np.random.default_rng(seed)

    # Build the generator list
    gens = []
    type_assignment = []  # which type group each gen belongs to (for diagnostics)
    for grp_idx, (count, type_name) in enumerate(fleet_spec):
        template = TYPE_TEMPLATES[type_name]
        for _ in range(count):
            g = _sample_unit(template, rng, plateau_factor=plateau_factor)
            gens.append(g)
            type_assignment.append(grp_idx)

    n_gen = len(gens)

    # Stack into arrays
    pmax = np.array([g['pmax'] for g in gens])
    pmin = np.array([g['pmin'] for g in gens])
    a_cost = np.array([g['a_cost'] for g in gens])
    b_cost = np.array([g['b_cost'] for g in gens])
    c_cost = np.array([g['c_cost'] for g in gens])
    su_cost = np.array([g['su_cost'] for g in gens])
    sd_cost = np.array([g['sd_cost'] for g in gens])
    min_up = np.array([g['min_up'] for g in gens], dtype=int)
    min_dn = np.array([g['min_dn'] for g in gens], dtype=int)
    ramp = np.array([g['ramp'] for g in gens])

    # ---- Demand: tight enough to force binding reserves but feasible ----
    # Peak at (1 - reserve_frac - 0.05) of capacity to leave clearance
    total_pmax = pmax.sum()
    capacity_factor = max(0.50, 1.0 - reserve_frac - 0.10)
    peak_demand = total_pmax * capacity_factor

    hours = np.arange(horizon)
    shape = 0.6 + 0.4 * np.sin((hours - 6) * 2 * np.pi / 24 - np.pi / 2)
    shape = np.clip(shape, 0.55, 1.0)
    shape *= 1 + 0.02 * rng.standard_normal(horizon)
    demand = peak_demand * shape

    reserve = reserve_frac * demand

    # ---- Initial state ----
    initial_on = (rng.random(n_gen) < 0.5).astype(int)
    initial_uptime = np.where(initial_on == 1, min_up, 0)
    initial_dntime = np.where(initial_on == 0, min_dn, 0)

    # Compute how many groups have count >= 2 for symmetry score
    sym_score = sum(c * (c - 1) for c, _ in fleet_spec) / 2

    return {
        'n_gen': n_gen,
        'horizon': horizon,
        'pmax': pmax, 'pmin': pmin,
        'a_cost': a_cost, 'b_cost': b_cost, 'c_cost': c_cost,
        'su_cost': su_cost, 'sd_cost': sd_cost,
        'min_up': min_up, 'min_dn': min_dn,
        'ramp': ramp,
        'demand': demand, 'reserve': reserve,
        'initial_on': initial_on,
        'initial_uptime': initial_uptime,
        'initial_dntime': initial_dntime,
        'reserve_frac': reserve_frac,
        # Metadata for analysis
        'fleet_spec': fleet_spec,
        'type_assignment': type_assignment,
        'plateau_factor': plateau_factor,
        'symmetry_score': sym_score,
    }


# Predefined fleet recipes at different difficulty levels
FLEET_RECIPES = {
    # Baseline: diverse fleet, no two units alike (matches original generator)
    'diverse_20': [(6, 'baseload_med'), (10, 'midmerit'), (4, 'peaker')],

    # Mild symmetry: 4-unit groups
    'sym4_20': [(4, 'baseload_large'), (4, 'baseload_med'),
                (8, 'midmerit'), (4, 'peaker')],

    # Strong symmetry: ~half the fleet is in identical groups of 8
    'sym8_24': [(8, 'baseload_med'), (8, 'midmerit'), (8, 'peaker')],

    # Extreme symmetry: all identical baseload + identical peakers
    'sym_extreme_20': [(12, 'baseload_med'), (8, 'peaker')],

    # Larger versions for scaling
    'sym8_40': [(16, 'baseload_med'), (16, 'midmerit'), (8, 'peaker')],
    'sym8_60': [(24, 'baseload_med'), (24, 'midmerit'), (12, 'peaker')],
}


if __name__ == '__main__':
    print("=== Hard UC instance generator smoke test ===\n")
    for name, recipe in FLEET_RECIPES.items():
        for plateau in [1.0, 0.1]:
            inst = generate_hard_uc(recipe, horizon=24, reserve_frac=0.15,
                                    plateau_factor=plateau, seed=42)
            print(f"{name:18s} plateau={plateau:.1f}: "
                  f"n_gen={inst['n_gen']:3d}, "
                  f"sym_score={inst['symmetry_score']:5.0f}, "
                  f"total_cap={inst['pmax'].sum():7.0f} MW, "
                  f"peak_demand={inst['demand'].max():6.0f} MW")
