# UC Experiment V5 — Transmission via Egret

Network-size sweep over four electrical networks of increasing size, each
solved with full DC-SCUC (commitment + dispatch + transmission via PTDF).
Measures how branching and wall-clock scale with `n_branch`.

| network | n_bus | n_branch | n_gen | horizon | source |
|---|---:|---:|---:|---:|---|
| case30   |  30 |  41 |  6 | 24 | pypower MATPOWER + synth UC params |
| RTS-GMLC |  73 | 120 | 156 | 24 | github.com/GridMod/RTS-GMLC (real) |
| IEEE-118 | 118 | 186 |  54 | 12 | pypower + synth UC |
| case300  | 300 | 411 |  69 |  8 | pypower + synth UC |

## Setup (one-time)

```bash
pip install gridx-egret pyomo pypower highspy
# Optional: SCIP for the second solver
pip install pyscipopt

# Real RTS-GMLC data (~30 MB)
mkdir -p uc_experiment_v5/data
git clone --depth 1 https://github.com/GridMod/RTS-GMLC.git uc_experiment_v5/data/RTS-GMLC
```

If RTS-GMLC isn't cloned, the loader raises FileNotFoundError; the driver
records that as an error row and continues with the other networks.

## Files

| file | purpose |
|---|---|
| `egret_loader.py` | Per-network `load_<name>(horizon, seed)` -> Egret ModelData; uniform `build_pyomo_scuc(md)`. |
| `pyomo_solver_bridge.py` | Pyomo -> MPS file -> highspy / pyscipopt. Returns v3-shaped result dict (with `mip_node_count`). |
| `v5_common.py` | Multiprocessing worker; re-imports v3's `checkpoint_csv` + `force_single_thread_env`. |
| `run_phase1_network_sweep.py` | Sweep driver. |
| `run_v5_all.py` | Top-level dispatcher. |
| `analyze_v5.py` | Log-log scaling fits + per-solver bar charts. |
| `v5_findings.md` | Write-up template. |

## Running

Smoke test (a few minutes, case30+RTS-GMLC at T=6):
```bash
python uc_experiment_v5/run_v5_all.py --quick --workers 2
```

Full sweep (all 4 networks, HiGHS + SCIP if installed):
```bash
python uc_experiment_v5/run_v5_all.py
```

Network subset:
```bash
python uc_experiment_v5/run_v5_all.py --networks case30 rts_gmlc
```

Re-analyse without re-solving:
```bash
python uc_experiment_v5/run_v5_all.py --analyze-only
```

## Notes

- **Lossy bits in the loader.** MATPOWER cases (case30, IEEE-118, case300) are
  single-period OPF datasets. UC parameters (min_up, min_dn, ramps, startup
  costs) are *synthesised* by a heuristic keyed on `Pmax`. The load profile
  is a fixed daily curve with small per-seed noise. Objective values from
  these synthetic UC instances are not comparable to any reference; their
  job in V5 is to provide a network-size axis with realistic topology.
  RTS-GMLC is the only network with real UC data.
- **Pyomo overhead is in the timings.** Each cell builds the Pyomo model
  then writes an MPS file before HiGHS/SCIP sees it. For small networks
  the overhead can rival the MIP solve itself; that's flagged on every
  plot.
- **SCIP graceful degradation.** Same pattern as v3 phase 4 / v3 phase 5:
  if `pyscipopt` isn't installed, the driver warns and runs HiGHS only.
- The bridge writes MPS rather than using `SolverFactory('appsi_highs')`
  because APPSI's HiGHS interface does not surface `mip_node_count`, which
  is the metric the analysis cares about most.
