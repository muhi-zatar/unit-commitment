# UC Empirical Experiment — Hard Instances

Companion to the SE/UC empirical validation. This package contains an
extended UC experiment designed to **force the MILP solver to actually
branch**, building on the original UC experiment where every instance
solved at the root node (`N_nodes = 1`).

## What's here

```
uc_experiment_v2/
├── README.md                  ← this file
├── uc_formulations.py         ← from original experiment; three formulations
├── uc_solver.py               ← from original experiment; basic solve_uc()
├── uc_hard_instance.py        ← NEW: parameterized hard-instance generator
├── instance_adapter.py        ← NEW: format converter
├── uc_hard_solver.py          ← NEW: solver with controllable HiGHS options
├── diagnose_branching.py      ← NEW: targeted instances to diagnose branching
└── (run scripts you'll add)
```

## Setup

```bash
pip install --break-system-packages numpy pandas matplotlib highspy scipy
```

Tested with HiGHS 1.14.0, NumPy 2.4, Python 3.12.

## Quick smoke test

```bash
# Generate a few hard instances, see their structural metrics
python uc_hard_instance.py

# Solve a hard instance with the basic solver
python instance_adapter.py

# Compare solver settings on a symmetric fleet
python uc_hard_solver.py

# Diagnose what creates branching: targeted demand patterns
python diagnose_branching.py
```

## What we found in the previous turn

When we ran the original UC experiment, **every single instance solved
in 1 branch-and-bound node** thanks to HiGHS's presolve and primal
heuristics. The exponential branching machinery from the master cost
expression was never exercised.

This package extends the experiment with three knobs designed to force
branching:

1. **Symmetric fleets** (`uc_hard_instance.py` → `FLEET_RECIPES`)
   Multiple identical generators create equivalent fractional LP solutions
   that should defeat heuristics.

2. **Cost plateau** (`plateau_factor` in `generate_hard_uc()`)
   Compresses the random spread within each generator type, making units
   nearly indistinguishable.

3. **Controlled solver settings** (`uc_hard_solver.py`)
   Exposes `presolve`, `mip_heuristic_effort`, and `mip_detect_symmetry`
   so you can disable HiGHS's defenses one at a time.

## What I observed before sending it to you

- Symmetric instances (`sym8_24`) produced LP gaps of **2-3%** (vs. <1%
  for the original generator) — this confirms tight gaps were a feature
  of the diverse-fleet design.
- **But still 1 node**, even with `detect_symmetry=False`,
  `heuristics_effort=0.0`, and `presolve='off'`.
- **HiGHS does not fully expose primal-heuristic disable**, so even with
  `mip_heuristic_effort=0.0` it generates a heuristic integer-feasible
  solution that closes the gap at the root.

The LP relaxation IS fractional (we verified u values like 0.30, 0.45 in
the LP solution), so branching SHOULD happen. But HiGHS's root-node
processing wraps it up before any branching occurs.

## Hypotheses to test locally

Things you can try with a tight iteration loop:

1. **Massive fleets**: try `n_identical = 100, 200, 500` — at some point
   the per-node LP becomes expensive enough to make the branching
   observable even if the tree is small.

2. **Time limit + node limit**: set `mip_max_nodes = 0` to force the
   solver to report the root LP solution + first few branching decisions
   without solving to optimality. This exposes the underlying difficulty
   even when default HiGHS would close it.

3. **Try a different solver**: install `pulp` or `pyomo` and use CBC
   (which has weaker heuristics than HiGHS). CBC will branch on these
   instances:
   ```bash
   pip install pulp  # bundles CBC
   ```

4. **PGLib-UC**: the optimization community's benchmark instances are
   designed to be hard. Download from https://github.com/power-grid-lib/pglib-uc
   and adapt them to feed `to_formulation_format()`.

5. **Symmetry-breaking constraints SHOULD make it easier, not harder**:
   add explicit ordering constraints on identical units (like
   `u[i, t] >= u[i+1, t]` for identical units i, i+1 in the same group).
   Compare runtimes — if symmetry-breaking constraints help, that
   confirms the LP-with-symmetry was the bottleneck even though we
   couldn't see branching.

## Suggested experiment for once branching IS visible

Once you have a configuration that produces `n_nodes >> 1`, the natural
sweep is:

```python
# In a new run_hard_sweep.py:
fleet_recipes = ['diverse_20', 'sym4_20', 'sym8_24', 'sym_extreme_20',
                 'sym8_40', 'sym8_60']
plateau_factors = [1.0, 0.5, 0.1, 0.01]  # how perfectly identical
formulations = ['three_bin', 'three_bin_tight']
seeds = [42, 59, 73]  # 3 replicates

# Measure how n_nodes scales with symmetry_score and plateau_factor.
# The empirical finding will be N_nodes ~ exp(symmetry_score * (1 - plateau_factor))
# or something similar — that's the actual exponential blow-up the master
# expression predicts.
```

## Module reference

### `uc_hard_instance.py`

```python
from uc_hard_instance import generate_hard_uc, FLEET_RECIPES

# Recipe = list of (count, type_name) tuples
recipe = [(8, 'baseload_med'), (8, 'midmerit'), (8, 'peaker')]
inst = generate_hard_uc(recipe, horizon=24,
                        reserve_frac=0.15,
                        plateau_factor=0.1,  # 0=identical, 1=full diversity
                        seed=42)
# inst is a dict with 'n_gen', 'pmax', 'demand', etc.
# Plus 'symmetry_score' for diagnostics.
```

Available types: `baseload_large`, `baseload_med`, `midmerit`, `peaker`.

### `instance_adapter.py`

```python
from instance_adapter import to_formulation_format
inst_for_solver = to_formulation_format(inst)  # converts field names
```

Needed because `uc_formulations.py` uses different field names
(`P_max`, `cost_a`, etc.) than the new generator (`pmax`, `a_cost`).

### `uc_hard_solver.py`

```python
from uc_hard_solver import solve_uc_with_settings
result = solve_uc_with_settings(
    inst_for_solver,
    formulation='three_bin',          # or 'three_bin_tight', 'perspective'
    presolve='on',                    # 'on' or 'off'
    heuristics_effort=0.05,           # 0.0 to 1.0
    detect_symmetry=True,             # True or False
    time_limit=120.0,
    gap_tol=0.001,
)
# Returns dict with mip_total_time_s, mip_node_count, root_lp_gap_pct, etc.
```

### `diagnose_branching.py`

Standalone script with a few targeted experiments. Reads as documentation
of what we tried and what didn't work.

## Notes on the original experiment

The data and findings from the original UC experiment (where everything
solved in 1 node) are in `../uc_experiment/`. The key files there:

- `uc_results.csv` — 60 runs across (5 sizes × 2 horizons × 3 formulations × 2 reps)
- `uc_empirical_validation.pdf` — 8-page write-up of those findings
- `plots/` — 6 scaling plots

That experiment is complete and self-contained; this v2 package is the
extension to find configurations that actually branch.

## What to send back

Once you have results, the analysis script from the original experiment
should work mostly as-is on a CSV with the same columns. If you change
the column structure, update `analyze_uc.py` accordingly.

The most interesting findings to look for:

1. **Does any setting produce `n_nodes > 1`?** If so, what's the
   relationship between symmetry_score and node count?
2. **Does CBC (via PuLP) branch where HiGHS doesn't?** Same problems,
   different solver behavior.
3. **What's the actual time scaling when branching does happen?**
   If `n_nodes ~ N^gamma`, we'd expect total_time ~ N^(gamma + 1).
