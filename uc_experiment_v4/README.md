# UC Experiment V4 — Formulation Fidelity Ablation

Full 2^4 factorial over four UC formulation features the v3 baseline ignored:

| flag | feature |
|---|---|
| F1 | piecewise quadratic cost (4 segments) |
| F2 | separate startup / shutdown ramp limits |
| F3 | lag-dependent startup costs (hot/warm/cold brackets) |
| F4 | must-run enforcement |

All 16 combinations are run on **two instance sets**: V3's synthetic
symmetry-sweep instances (extended to carry the new fields) and a small
subset of PGLib-UC (which already carries them natively).

## Files

| file | purpose |
|---|---|
| [uc_formulations_v4.py](../uc_formulations_v4.py) (parent dir) | Extended `build_three_bin_v4(inst, features)`. |
| `uc_hard_solver_v4.py` | Solver wrapper; mirrors v3's `solve_uc_with_settings`. |
| `symmetry_sweep_instances_v4.py` | Synthetic generator with v4 fields added. |
| `pglib_uc_adapter_v4.py` | Non-lossy PGLib adapter; refits quadratic cost from breakpoints. |
| `v4_common.py` | Multiprocessing workers; re-uses v3's `checkpoint_csv` + `force_single_thread_env`. |
| `run_phase1_synth_v4.py` | Synth driver. |
| `run_phase2_pglib_v4.py` | PGLib driver. |
| `run_v4_all.py` | Top-level dispatcher. |
| `analyze_v4.py` | 2^4 factorial regression + plots. |
| `v4_findings.md` | Write-up template. |

## Running

Smoke test (a few minutes):
```bash
python uc_experiment_v4/run_v4_all.py --quick --workers 2
```

Full sweep:
```bash
python uc_experiment_v4/run_v4_all.py
```

Phase 2 needs PGLib-UC at `uc_experiment_v3/pglib-uc/` (run v3's
`fetch_pglib_uc.py` first), or pass `--pglib-root <path>`.

Re-analyse without re-solving:
```bash
python uc_experiment_v4/run_v4_all.py --analyze-only
```

## Notes

- V4 keeps the v3 file layout: per-phase CSV next to the driver, `plots/` for
  figures, atomic checkpoint via `v3_common.checkpoint_csv`.
- SCIP is not exercised in V4. The factorial ablation only needs HiGHS, since
  it isolates **formulation effects**, not solver effects. A future v4b could
  run the same factorial under SCIP.
- F4 (must-run) on the synthetic instances marks ~20% of the smallest-pmin
  units in the *diverse block only* — never the identical block, which would
  collapse the symmetry signal v3 cares about.
- PGLib instances rarely carry `must_run=1`; expect F4 to be a near-no-op on
  the PGLib subset unless you hand-flag some units.
