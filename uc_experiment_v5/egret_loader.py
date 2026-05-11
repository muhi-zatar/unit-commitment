"""
V5 loaders: per-network SCUC ModelData builders, plus a uniform Pyomo
SCUC build wrapper.

Strategy:
  - RTS-GMLC:  use Egret's bundled parser; this is a real SCUC dataset
               with hourly load/reserve time series.
  - MATPOWER cases (IEEE-118, Polish 2383): use pypower for the network
               (bus, branch, gen) and synthesize UC fields (min_up, min_dn,
               ramp_up, ramp_dn, startup_cost, no_load_cost) plus an hourly
               load profile from a generic daily-curve shape.

The returned object is always an Egret ModelData. The uniform
`build_pyomo_scuc(md)` wraps `create_tight_unit_commitment_model` with
PTDF transmission.
"""

import os
import sys
import warnings

import numpy as np
import pyomo.environ  # noqa: F401  ensures pyomo plugins are registered

from egret.data.model_data import ModelData
from egret.parsers.rts_gmlc.parser import create_ModelData as _rts_create_ModelData
from egret.models.unit_commitment import create_tight_unit_commitment_model


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RTS_GMLC_PATH = os.path.join(
    THIS_DIR, 'data', 'RTS-GMLC', 'RTS_Data', 'SourceData'
)


# ---------- RTS-GMLC -------------------------------------------------------

def load_rts_gmlc(horizon=24, seed=42,
                  rts_path=None, begin='2020-01-27', end=None):
    """
    Build an RTS-GMLC SCUC ModelData. Egret's parser handles all the heavy
    lifting; we just clamp the resulting time series to `horizon` hours.

    `seed` is ignored (RTS-GMLC has fixed data) but kept for API symmetry.
    """
    if rts_path is None:
        rts_path = DEFAULT_RTS_GMLC_PATH
    if not os.path.isdir(rts_path):
        raise FileNotFoundError(
            f"RTS-GMLC data not found at {rts_path}. "
            f"Clone github.com/GridMod/RTS-GMLC there or pass --rts-path."
        )
    if end is None:
        # Day after `begin` (Egret's parser is half-open on the end).
        from datetime import datetime, timedelta
        d0 = datetime.strptime(begin, '%Y-%m-%d')
        end = (d0 + timedelta(days=1)).strftime('%Y-%m-%d')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        md = _rts_create_ModelData(rts_path, begin, end, simulation='DAY_AHEAD')

    # RTS-GMLC ships 24 hours per day at DAY_AHEAD; truncate further if asked
    if horizon and horizon < len(md.data['system']['time_keys']):
        _truncate_time_series(md, horizon)
    return md


# ---------- MATPOWER + synthesised UC --------------------------------------

# Daily load shape (one value per hour). Calibrated very loosely to look
# like a typical North American summer day; baseline 0.6, peak 1.0.
_DAILY_SHAPE = np.array([
    0.62, 0.59, 0.57, 0.56, 0.56, 0.58, 0.63, 0.70,
    0.78, 0.84, 0.88, 0.91, 0.93, 0.95, 0.97, 0.99,
    1.00, 1.00, 0.97, 0.93, 0.88, 0.81, 0.74, 0.68,
])


# Heuristic UC params keyed off Pmax. Tuned to leave headroom for the
# SCUC to remain feasible while still exercising commitment decisions.
def _synth_uc_params_for_unit(pmax, rng):
    if pmax > 400:
        kind = 'baseload_large'
        return dict(min_up=rng.integers(6, 10), min_dn=rng.integers(6, 10),
                    ramp_frac=0.30, pmin_frac=0.40, fuel_b=20.0, fuel_c=200.0,
                    su_cost=10000.0)
    if pmax > 150:
        kind = 'baseload_med'
        return dict(min_up=rng.integers(4, 7), min_dn=rng.integers(4, 7),
                    ramp_frac=0.40, pmin_frac=0.35, fuel_b=25.0, fuel_c=120.0,
                    su_cost=4000.0)
    if pmax > 50:
        kind = 'midmerit'
        return dict(min_up=rng.integers(2, 5), min_dn=rng.integers(2, 5),
                    ramp_frac=0.55, pmin_frac=0.30, fuel_b=35.0, fuel_c=80.0,
                    su_cost=2000.0)
    return dict(min_up=1, min_dn=1,
                ramp_frac=0.80, pmin_frac=0.20, fuel_b=60.0, fuel_c=40.0,
                su_cost=400.0)


def _model_data_from_pypower(case, horizon, seed, name):
    """
    Build an Egret ModelData from a pypower case dict + synthesised UC fields.

    case: dict with arrays bus, branch, gen (pypower format).
    """
    rng = np.random.default_rng(seed)

    md = ModelData()
    md.data['system']['name'] = name
    md.data['system']['baseMVA'] = float(case.get('baseMVA', 100.0))
    # Time keys: T strings 'T01', 'T02', ...
    time_keys = [f'T{t+1:02d}' for t in range(horizon)]
    md.data['system']['time_keys'] = time_keys
    md.data['system']['time_period_length_minutes'] = 60

    elements = md.data['elements']
    elements['bus'] = {}
    elements['branch'] = {}
    elements['generator'] = {}
    elements['load'] = {}
    elements['area'] = {'area_1': {}}

    # Buses
    bus_array = case['bus']
    bus_id_to_name = {}
    base_voltage = 230.0
    for row in bus_array:
        bus_id = int(row[0])
        name = f'b{bus_id}'
        bus_id_to_name[bus_id] = name
        elements['bus'][name] = {
            'matpower_bustype': int(row[1]),
            'va': 0.0,
            'vm': float(row[7]) if row[7] > 0 else 1.0,
            'base_kv': float(row[9]) if row[9] > 0 else base_voltage,
            'area': 'area_1',
            'in_service': True,
            'v_min': 0.95, 'v_max': 1.05,
        }

    # Loads (one constant Pload per bus, scaled by daily shape across T)
    daily = np.tile(_DAILY_SHAPE, horizon // 24 + 1)[:horizon]
    daily = daily * (1 + 0.02 * rng.standard_normal(horizon))
    daily = np.clip(daily, 0.5, 1.05)
    for row in bus_array:
        bus_id = int(row[0])
        Pd = float(row[2])
        Qd = float(row[3])
        if abs(Pd) < 1e-6 and abs(Qd) < 1e-6:
            continue
        bus_name = bus_id_to_name[bus_id]
        load_name = f'l_{bus_name}'
        elements['load'][load_name] = {
            'bus': bus_name,
            'in_service': True,
            'p_load': {'data_type': 'time_series',
                       'values': [float(Pd * d) for d in daily]},
            'q_load': {'data_type': 'time_series',
                       'values': [float(Qd * d) for d in daily]},
        }

    # Branches
    for i, row in enumerate(case['branch']):
        from_id = int(row[0]); to_id = int(row[1])
        name = f'br{i}'
        rate = float(row[5])
        if rate <= 0:
            rate = 9999.0
        elements['branch'][name] = {
            'from_bus': bus_id_to_name[from_id],
            'to_bus': bus_id_to_name[to_id],
            'resistance': float(row[2]),
            'reactance': float(row[3]) if row[3] != 0 else 1e-4,
            'charging_susceptance': float(row[4]),
            'rating_long_term': rate,
            'rating_short_term': rate,
            'rating_emergency': rate,
            'transformer_tap_ratio': float(row[8]) if row[8] != 0 else 1.0,
            'transformer_phase_shift': float(row[9]),
            'in_service': bool(int(row[10])),
            'branch_type': 'line',
            'angle_diff_min': -90.0, 'angle_diff_max': 90.0,
            'pf': None, 'qf': None, 'pt': None, 'qt': None,
        }

    # Generators (thermal only)
    for i, row in enumerate(case['gen']):
        bus_id = int(row[0])
        pmax = float(row[8])
        if pmax <= 0:
            continue
        name = f'g{i}'
        prm = _synth_uc_params_for_unit(pmax, rng)
        pmin = pmax * prm['pmin_frac']
        ramp = pmax * prm['ramp_frac']
        # Initial state: on if the OPF case had this gen running (Pg > 0)
        init_pg = float(row[1])
        unit_on_t0 = 1 if init_pg > 1e-3 else 0
        # cost curve: linear b * P + no-load c (matches three_bin n_pwl=1 v3 style)
        b_cost = float(prm['fuel_b'])
        no_load = float(prm['fuel_c'])
        elements['generator'][name] = {
            'bus': bus_id_to_name[bus_id],
            'generator_type': 'thermal',
            'fuel': 'NG' if pmax < 200 else 'COAL',
            'in_service': True,
            'p_min': float(pmin),
            'p_max': float(pmax),
            'q_min': float(row[4]),
            'q_max': float(row[3]),
            'ramp_up_60min': float(ramp),
            'ramp_down_60min': float(ramp),
            'startup_capacity': float(max(pmin * 1.05, ramp * 0.5)),
            'shutdown_capacity': float(max(pmin * 1.05, ramp * 0.5)),
            'min_up_time': int(prm['min_up']),
            'min_down_time': int(prm['min_dn']),
            'initial_status': int(unit_on_t0 * prm['min_up'] +
                                  (1 - unit_on_t0) * (-prm['min_dn'])),
            'initial_p_output': float(init_pg) if unit_on_t0 else 0.0,
            'startup_cost': [(int(prm['min_dn']), float(prm['su_cost'])),
                             (int(prm['min_dn'] + 4), float(prm['su_cost'] * 1.5))],
            'shutdown_cost': 0.0,
            'p_cost': {
                'data_type': 'cost_curve',
                'cost_curve_type': 'piecewise',
                'values': [(float(pmin), float(pmin * b_cost + no_load)),
                           (float(pmax), float(pmax * b_cost + no_load))],
            },
            'must_run': False,
            'fast_start': pmax < 50,
        }

    # System-level reserve requirement: 5% of peak load
    total_peak = sum(
        max(L['p_load']['values']) for L in elements['load'].values()
    ) if elements['load'] else 1.0
    md.data['system']['reserve_requirement'] = {
        'data_type': 'time_series',
        'values': [float(0.05 * total_peak * d) for d in daily],
    }

    return md


def _truncate_time_series(md, horizon):
    """In-place truncate all time series fields to the first `horizon` entries."""
    md.data['system']['time_keys'] = md.data['system']['time_keys'][:horizon]
    for section in ('load', 'generator'):
        for elt in md.data['elements'].get(section, {}).values():
            for k, v in list(elt.items()):
                if isinstance(v, dict) and v.get('data_type') == 'time_series':
                    elt[k] = {'data_type': 'time_series',
                              'values': v['values'][:horizon]}
    sysrr = md.data['system'].get('reserve_requirement')
    if isinstance(sysrr, dict) and sysrr.get('data_type') == 'time_series':
        md.data['system']['reserve_requirement']['values'] = \
            sysrr['values'][:horizon]


def load_ieee118(horizon=24, seed=42):
    import pypower.api as pp
    case = pp.case118()
    return _model_data_from_pypower(case, horizon, seed, 'ieee118')


def load_case30(horizon=24, seed=42):
    import pypower.api as pp
    case = pp.case30()
    return _model_data_from_pypower(case, horizon, seed, 'case30')


def load_case300(horizon=24, seed=42):
    import pypower.api as pp
    case = pp.case300()
    return _model_data_from_pypower(case, horizon, seed, 'case300')


NETWORK_LOADERS = {
    'case30':   load_case30,
    'rts_gmlc': load_rts_gmlc,
    'ieee118':  load_ieee118,
    'case300':  load_case300,
}


# ---------- Uniform SCUC build ---------------------------------------------

def build_pyomo_scuc(md, network_constraints='ptdf_power_flow'):
    """Wrap Egret's tight UC builder with default options."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = create_tight_unit_commitment_model(
            md, network_constraints=network_constraints
        )
    return model


def network_size(md):
    """Return (n_bus, n_branch, n_gen, T) — used for the scaling sweep."""
    elts = md.data['elements']
    return (
        len(elts.get('bus', {})),
        len(elts.get('branch', {})),
        len(elts.get('generator', {})),
        len(md.data['system'].get('time_keys', [])),
    )


if __name__ == '__main__':
    network = sys.argv[1] if len(sys.argv) > 1 else 'rts_gmlc'
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    print(f'>>> loading {network} (horizon={horizon}) ...', flush=True)
    md = NETWORK_LOADERS[network](horizon=horizon)
    n_bus, n_br, n_gen, T = network_size(md)
    print(f'    n_bus={n_bus} n_branch={n_br} n_gen={n_gen} T={T}')

    print('>>> building Pyomo SCUC...', flush=True)
    model = build_pyomo_scuc(md)
    n_bin = sum(1 for v in model.component_data_objects()
                if hasattr(v, 'is_binary') and v.is_binary())
    print(f'    n_var={model.nvariables()} '
          f'n_constr={model.nconstraints()} '
          f'n_binary={n_bin}')
