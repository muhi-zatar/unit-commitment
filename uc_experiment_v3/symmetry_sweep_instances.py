"""
v3 instance generator: continuous symmetry parameter.

Replaces v2's discrete {diverse, sym4, sym8} axis with a single integer k.
N_g total units; k of them are bit-for-bit identical; the remaining N_g - k
are sampled with full diversity (plateau_factor=1.0).

k = 1   -> fully diverse fleet (no two units identical)
k = N_g -> all N_g units identical

The k identical units are chosen from a single TYPE_TEMPLATE (default
'midmerit', the middle of the merit order) so the identical block is a
plausible plant-cluster and the sym pressure isn't trivially absorbed by
the merit order pushing the cluster to the rails.
"""

import os
import sys

import numpy as np

# Allow running as `python -m uc_experiment_v3.X` or as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from uc_hard_instance import TYPE_TEMPLATES, _sample_unit  # noqa: E402


DEFAULT_IDENTICAL_TYPE = 'midmerit'


def make_symmetry_sweep_instance(N_g, k_identical, horizon, seed,
                                 reserve_frac=0.15,
                                 identical_type=DEFAULT_IDENTICAL_TYPE):
    """
    Build a UC instance with a controlled symmetry parameter.

    Parameters
    ----------
    N_g : int
        Total number of generators.
    k_identical : int in [1, N_g]
        How many units are bit-for-bit identical. The remaining N_g-k
        units are drawn from TYPE_TEMPLATES with full diversity.
    horizon : int
        Time horizon in hours.
    seed : int
        RNG seed.
    reserve_frac : float
        Spinning reserve fraction of demand.
    identical_type : str
        Which TYPE_TEMPLATE to use for the identical block.

    Returns a dict with the same shape as generate_hard_uc, with:
        symmetry_score = k_identical * (k_identical - 1) / 2
    """
    if k_identical < 1 or k_identical > N_g:
        raise ValueError(f"k_identical must be in [1, {N_g}], got {k_identical}")
    if identical_type not in TYPE_TEMPLATES:
        raise ValueError(f"unknown identical_type {identical_type!r}")

    rng = np.random.default_rng(seed)

    # Build the identical block first: one prototype unit, then replicated.
    proto_rng = np.random.default_rng(seed ^ 0xA5A5)
    proto = _sample_unit(TYPE_TEMPLATES[identical_type], proto_rng,
                         plateau_factor=0.0)  # midpoint, fully deterministic

    gens = []
    type_assignment = []
    for _ in range(k_identical):
        gens.append(dict(proto))  # copy so downstream mutation is safe
        type_assignment.append(0)  # group 0 = identical block

    # Build the diverse remainder: sample with plateau=1.0.
    # Match v2's distribution (baseload_med / midmerit / peaker): baseload_large
    # is excluded because its pmin_frac > ramp_frac means off units can't be
    # started in one step, which makes some demand-trough hours infeasible.
    n_remainder = N_g - k_identical
    if n_remainder > 0:
        weights = {
            'baseload_med': 0.30,
            'midmerit':     0.50,
            'peaker':       0.20,
        }
        # Deterministic split: floor then top up the largest bucket.
        counts = {t: int(np.floor(w * n_remainder)) for t, w in weights.items()}
        leftover = n_remainder - sum(counts.values())
        # Distribute leftover to types in stable order
        for t in sorted(weights, key=lambda x: -weights[x]):
            if leftover <= 0:
                break
            counts[t] += 1
            leftover -= 1

        grp_idx = 1
        for type_name, cnt in counts.items():
            if cnt <= 0:
                continue
            tmpl = TYPE_TEMPLATES[type_name]
            for _ in range(cnt):
                gens.append(_sample_unit(tmpl, rng, plateau_factor=1.0))
                type_assignment.append(grp_idx)
            grp_idx += 1

    assert len(gens) == N_g

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

    # ---- Demand: same logic as generate_hard_uc ----
    total_pmax = pmax.sum()
    capacity_factor = max(0.50, 1.0 - reserve_frac - 0.10)
    peak_demand = total_pmax * capacity_factor

    hours = np.arange(horizon)
    shape = 0.6 + 0.4 * np.sin((hours - 6) * 2 * np.pi / 24 - np.pi / 2)
    shape = np.clip(shape, 0.55, 1.0)
    shape *= 1 + 0.02 * rng.standard_normal(horizon)
    demand = peak_demand * shape
    reserve = reserve_frac * demand

    initial_on = (rng.random(N_g) < 0.5).astype(int)
    initial_uptime = np.where(initial_on == 1, min_up, 0)
    initial_dntime = np.where(initial_on == 0, min_dn, 0)

    sym_score = k_identical * (k_identical - 1) / 2

    # Sparse fleet_spec for diagnostics; analysis only reads symmetry_score/k.
    fleet_spec = [(k_identical, identical_type)]

    return {
        'n_gen': N_g,
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
        # Metadata
        'fleet_spec': fleet_spec,
        'type_assignment': type_assignment,
        'plateau_factor': None,    # not meaningful here; mixed within instance
        'symmetry_score': sym_score,
        'k_identical': k_identical,
        'N_g': N_g,
    }


def instance_id(N_g, k_identical, horizon, seed):
    """Canonical id string for use as a CSV key."""
    return f"Ng{N_g}_k{k_identical}_T{horizon}_s{seed}"


if __name__ == '__main__':
    print("=== v3 symmetry-sweep instance generator smoke test ===\n")
    for N_g in (24, 48):
        for k in (1, N_g // 2, N_g):
            inst = make_symmetry_sweep_instance(N_g, k, horizon=24, seed=42)
            print(f"N_g={N_g:3d} k={k:3d} : "
                  f"sym_score={inst['symmetry_score']:6.0f}, "
                  f"total_cap={inst['pmax'].sum():7.0f} MW, "
                  f"peak_demand={inst['demand'].max():6.0f} MW")
