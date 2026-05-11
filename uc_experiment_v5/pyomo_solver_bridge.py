"""
Bridge: Pyomo ConcreteModel  ->  MPS file  ->  highspy or pyscipopt.

Why MPS-via-file and not pyomo.opt.SolverFactory? Because the APPSI Highs
result object does NOT surface node count, and that is the primary metric
v3/v4 measure. Writing MPS and re-reading it in highspy/pyscipopt lets us
use the same `getInfo().mip_node_count` / `getNNodes()` v3 already
relies on, at the cost of a few seconds of I/O.

The returned dict matches v3's result-dict shape (mip_total_time_s,
mip_node_count, mip_obj, mip_dual_bound, mip_gap, status, ...).
"""

import os
import sys
import tempfile
import time
import warnings


def _write_mps(pyomo_model):
    """Pyomo -> temp .mps file. Returns (path, write_time_s)."""
    with tempfile.NamedTemporaryFile(suffix='.mps', delete=False) as f:
        mps_path = f.name
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pyomo_model.write(mps_path, format='mps')
    return mps_path, time.perf_counter() - t0


def solve_pyomo_uc_highs(pyomo_model, time_limit=600.0, gap_tol=0.001,
                          presolve='on', heuristics_effort=0.05,
                          detect_symmetry=True):
    """Solve a Pyomo UC model via HiGHS by writing MPS first."""
    import highspy

    mps_path, write_time = _write_mps(pyomo_model)
    try:
        h = highspy.Highs(); h.silent()
        h.readModel(mps_path)
        h.setOptionValue('time_limit', float(time_limit))
        h.setOptionValue('mip_rel_gap', float(gap_tol))
        h.setOptionValue('output_flag', False)
        h.setOptionValue('presolve', presolve)
        h.setOptionValue('mip_heuristic_effort', float(heuristics_effort))
        h.setOptionValue('mip_detect_symmetry', bool(detect_symmetry))
        h.setOptionValue('threads', 1)

        t0 = time.perf_counter()
        h.run()
        mip_time = time.perf_counter() - t0
        mi = h.getInfo()
        status = h.modelStatusToString(h.getModelStatus())
        result = {
            'solver': 'highs',
            'pyomo_write_time_s': write_time,
            'mip_total_time_s': mip_time,
            'mip_obj': float(mi.objective_function_value),
            'mip_dual_bound': float(mi.mip_dual_bound),
            'mip_gap': float(mi.mip_gap),
            'mip_node_count': int(mi.mip_node_count),
            'mip_simplex_iters': int(mi.simplex_iteration_count),
            'status': status,
            'converged': ('optimal' in status.lower()
                          and float(mi.mip_gap) <= gap_tol * 1.01),
            'n_var': int(h.getNumCol()),
            'n_constr': int(h.getNumRow()),
            'presolve': presolve,
            'heuristics_effort': heuristics_effort,
            'detect_symmetry': detect_symmetry,
        }
    finally:
        try: os.unlink(mps_path)
        except OSError: pass
    return result


def solve_pyomo_uc_scip(pyomo_model, time_limit=600.0, gap_tol=0.001,
                         presolve='on', heuristics_effort=0.05,
                         detect_symmetry=True):
    """Solve a Pyomo UC model via SCIP by writing MPS first."""
    try:
        from pyscipopt import Model, SCIP_PARAMSETTING
    except ImportError:
        raise RuntimeError("pyscipopt not installed; cannot run SCIP")

    mps_path, write_time = _write_mps(pyomo_model)
    try:
        m = Model("V5_SCUC"); m.hideOutput()
        m.readProblem(mps_path)
        m.setParam('limits/time', float(time_limit))
        m.setParam('limits/gap', float(gap_tol))
        m.setParam('parallel/maxnthreads', 1)

        if presolve == 'off':
            m.setParam('presolving/maxrounds', 0)
            m.setParam('presolving/maxrestarts', 0)
        if heuristics_effort == 0.0:
            try:
                m.setHeuristics(SCIP_PARAMSETTING.OFF)
            except Exception:
                pass
        if not detect_symmetry:
            try:
                m.setParam('misc/usesymmetry', 0)
            except Exception:
                pass

        t0 = time.perf_counter()
        m.optimize()
        mip_time = time.perf_counter() - t0

        status = m.getStatus()
        try: obj = float(m.getObjVal())
        except Exception: obj = float('nan')
        try: dual = float(m.getDualbound())
        except Exception: dual = float('nan')
        try: gap = float(m.getGap())
        except Exception: gap = float('nan')

        result = {
            'solver': 'scip',
            'pyomo_write_time_s': write_time,
            'mip_total_time_s': mip_time,
            'mip_obj': obj,
            'mip_dual_bound': dual,
            'mip_gap': gap,
            'mip_node_count': int(m.getNNodes()),
            'mip_simplex_iters': int(m.getNLPIterations()),
            'status': status,
            'converged': status == 'optimal',
            'n_var': None, 'n_constr': None,
            'presolve': presolve,
            'heuristics_effort': heuristics_effort,
            'detect_symmetry': detect_symmetry,
        }
    finally:
        try: os.unlink(mps_path)
        except OSError: pass
    return result


def solve_pyomo_uc(pyomo_model, solver='highs', **kwargs):
    if solver == 'highs':
        return solve_pyomo_uc_highs(pyomo_model, **kwargs)
    if solver == 'scip':
        return solve_pyomo_uc_scip(pyomo_model, **kwargs)
    raise ValueError(f"unknown solver {solver!r}")


if __name__ == '__main__':
    # Trivial sanity-check: build a 2-var MILP, solve with both solvers
    import pyomo.environ as pyo
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.Binary)
    m.y = pyo.Var(bounds=(0, 10))
    m.obj = pyo.Objective(expr=2*m.x + m.y, sense=pyo.minimize)
    m.c1 = pyo.Constraint(expr=m.x + m.y >= 3)

    for slv in ('highs', 'scip'):
        try:
            r = solve_pyomo_uc(m, solver=slv, time_limit=10)
            print(f"{slv:5s}: obj={r['mip_obj']:.3f} "
                  f"nodes={r['mip_node_count']} "
                  f"time={r['mip_total_time_s']:.3f}s "
                  f"status={r['status']}")
        except Exception as e:
            print(f"{slv:5s}: ERROR {type(e).__name__}: {e}")
