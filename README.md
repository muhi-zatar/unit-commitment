# Unit Commitment — Empirical Solver-Cost Study

A multi-version study of how a modern MILP solver handles the unit-commitment
problem. Each version stresses a different axis. Single-bus UC mostly solves
at the root node under HiGHS defaults; adding symmetry, fidelity features,
or transmission progressively re-exposes branching.

## Setup

```bash
pip install numpy pandas matplotlib scipy highspy
# Optional (used by phases that compare solvers)
pip install pyscipopt statsmodels pyomo gridx-egret pypower
```

---

## V1 — Baseline

**What:** First sweep. 60 runs over (5 sizes × 2 horizons × 3 formulations × 2 reps).
Established that HiGHS solves every instance at 1 node and wall time scales
as `n^β` with β ≈ 0.76–0.99.

**Files:** [uc_instance.py](uc_instance.py), [uc_formulations.py](uc_formulations.py),
[uc_solver.py](uc_solver.py), [run_sweep.py](run_sweep.py),
[analyze_uc.py](analyze_uc.py), [uc_results.csv](uc_results.csv).

**Run:**
```bash
python run_sweep.py
python analyze_uc.py
```

---

## V2 — Symmetric hard instances

**What:** Hand-crafted symmetric fleets (`diverse_20`, `sym4_20`, `sym8_24`, ...)
to force branching. GLPK is used because HiGHS's symmetry detection collapses
these to 1 node. Confirmed symmetric fleets at T=8 produce ~5·10⁴ B&C nodes.

**Files:** [uc_hard_instance.py](uc_hard_instance.py),
[uc_hard_solver.py](uc_hard_solver.py), [uc_glpk_solver.py](uc_glpk_solver.py),
[instance_adapter.py](instance_adapter.py), [run_v2_sweep.py](run_v2_sweep.py),
[analyze_v2.py](analyze_v2.py), [diagnose_branching.py](diagnose_branching.py),
[uc_v2_results.csv](uc_v2_results.csv).

**Run:**
```bash
python run_v2_sweep.py
python analyze_v2.py
```

---

## V3 — Stress-testing the branching story

**What:** Five phases that turn the v2 anecdote into a defensible story.

| Phase | Question |
|---|---|
| 1 | How does `N_nodes` scale with the symmetry parameter `k_identical`? |
| 2 | Does HiGHS's `detect_symmetry` flag actually help? |
| 3 | Which solver feature (presolve / heuristics / symmetry) hides branching? |
| 4 | Is the v2 story HiGHS-specific or does SCIP show the same pattern? |
| 5 | Same questions on PGLib-UC real benchmark instances. |

**Folder:** [uc_experiment_v3/](uc_experiment_v3/) — drivers, analyzers,
results CSVs, plots.

**Run:**
```bash
# Smoke (~2 min)
python uc_experiment_v3/run_v3_all.py --quick --workers 2

# Full
python uc_experiment_v3/run_v3_all.py --workers 8

# One phase only
python uc_experiment_v3/run_v3_all.py --phase 3

# Phase 5 needs PGLib-UC checkout first:
python uc_experiment_v3/fetch_pglib_uc.py
```

---

## V4 — Formulation-fidelity factorial

**What:** Full 2⁴ factorial over four UC features V3 ignored:

| flag | feature |
|---|---|
| F1 | piecewise quadratic cost (4 segments) |
| F2 | separate startup / shutdown ramp limits |
| F3 | lag-dependent startup costs (hot/warm/cold) |
| F4 | must-run enforcement |

Run on both V3 synthetic instances and PGLib-UC. Analysis fits an OLS
regression `log(nodes) ~ F1+F2+F3+F4 + 2-way interactions` to identify
which features actually move branching.

**Folder:** [uc_experiment_v4/](uc_experiment_v4/) plus the extended
builder [uc_formulations_v4.py](uc_formulations_v4.py) at the project root.

**Run:**
```bash
# Smoke (~2 min)
python uc_experiment_v4/run_v4_all.py --quick --workers 2

# Full
python uc_experiment_v4/run_v4_all.py --workers 8

# Just one phase
python uc_experiment_v4/run_v4_all.py --phase 1   # synth
python uc_experiment_v4/run_v4_all.py --phase 2   # pglib
```

---

## V5 — Transmission via Egret

**What:** SCUC on real electrical networks of increasing size. Adds DC
power flow + PTDF line limits via NREL's Egret library. Networks:

| network  | n_bus | n_branch | UC data |
|---|---:|---:|---|
| case30   |  30 |  41 | synthesised |
| RTS-GMLC |  73 | 120 | real |
| IEEE-118 | 118 | 186 | synthesised |
| case300  | 300 | 411 | synthesised |

Analysis includes a V3-vs-V5 "transmission delta" — same RTS-GMLC
generators, with vs without the network. Headline finding: adding
transmission takes HiGHS from 1 node / ~2 s to 2900+ nodes / 600 s.

**Folder:** [uc_experiment_v5/](uc_experiment_v5/).

**Setup (one-time):**
```bash
pip install pyomo gridx-egret pypower
mkdir -p uc_experiment_v5/data
git clone --depth 1 https://github.com/GridMod/RTS-GMLC.git uc_experiment_v5/data/RTS-GMLC
```

**Run:**
```bash
# Smoke (~30 s)
python uc_experiment_v5/run_v5_all.py --quick --workers 2

# Full (~10 min; RTS-GMLC at T=24 typically hits 600 s time limit)
python uc_experiment_v5/run_v5_all.py --workers 2

# Subset
python uc_experiment_v5/run_v5_all.py --networks rts_gmlc ieee118
```

---

## At a glance

| version | knob | takeaway |
|---|---|---|
| V1 | size | HiGHS solves everything at 1 node. |
| V2 | hand-crafted symmetry | Old solvers (GLPK) branch a lot. |
| V3 | continuous symmetry + ablations | HiGHS's presolve+heuristics hide the tree. |
| V4 | formulation fidelity | PWL cost is the dominant time effect; node count mostly unchanged. |
| V5 | DC transmission | This is what actually exposes the MIP. 1 node → 10²–10³. |

Each folder has its own `README.md` and `vN_findings.md` template for the
write-up.
