# UC Experiment v3

Stress-testing the v2 branching story with proper ablations and a second solver.

## Files

| file | purpose |
|---|---|
| `symmetry_sweep_instances.py` | `make_symmetry_sweep_instance(N_g, k_identical, T, seed)` — continuous-k generator built on top of v2's `TYPE_TEMPLATES`. |
| `v3_common.py` | Shared worker (`_highs_worker`), atomic CSV checkpointing, thread-pinning helper. |
| `run_phase1_symmetry_sweep.py` | Phase 1 driver. |
| `analyze_phase1.py` | Phase 1 plots + summary. |
| `run_phase2_detect_symmetry.py` | Phase 2 driver (picks top-decile hardest from Phase 1). |
| `analyze_phase2.py` | Phase 2 paired scatter. |
| `run_phase3_mode_ablation.py` | Phase 3 driver (5 configs, 5 seeds). |
| `analyze_phase3.py` | Phase 3 bar charts. |
| `uc_solver_scip.py` | SCIP wrapper exposing the same interface as `uc_hard_solver`. Builds three_bin in pyscipopt. |
| `run_phase4_solver_bakeoff.py` | Phase 4 driver (HiGHS vs SCIP). |
| `analyze_phase4.py` | Phase 4 scatter + median-ratio table. |
| `pglib_uc_adapter.py` | Load PGLib-UC JSON → formulation-format inst dict. |
| `fetch_pglib_uc.py` | Clone PGLib-UC repo into `./pglib-uc/`. |
| `run_phase5_pglib.py` | Phase 5 driver (HiGHS + SCIP on PGLib small/medium). |
| `analyze_phase5.py` | Phase 5 plots + per-dataset summary. |
| `run_v3_all.py` | Top-level driver. |
| `v3_findings.md` | Template for the final write-up. |
| `plots/` | Output figures. |

## Running

Smoke test (a few minutes):

```bash
python uc_experiment_v3/run_v3_all.py --quick --workers 2
```

Full sweep:

```bash
python uc_experiment_v3/run_v3_all.py
```

Actual wall-clock observed: **a few minutes total** on a modern workstation,
not the 4-10 h the plan doc estimated. HiGHS solves nearly every cell at
the root node (≪1 s per instance), so even the large grids in Phase 1 and
Phase 5 finish quickly. The 4-10 h figure in the original plan assumed
solves that hit branching pressure under default settings; that mostly
doesn't happen with HiGHS. Expect minutes, not hours.

Single phase:

```bash
python uc_experiment_v3/run_v3_all.py --phase 1
```

Re-analyze without re-solving:

```bash
python uc_experiment_v3/run_v3_all.py --analyze-only
```

## Notes

- Phase 4 needs `pip install pyscipopt`. If SCIP is not installed the driver
  prints a warning and writes a HiGHS-only `phase4_results.csv`.
- v2 files are not touched. The v3 folder imports `uc_hard_instance.py`,
  `uc_hard_solver.py`, `instance_adapter.py`, and `uc_formulations.py` from
  the parent directory.
- Each phase checkpoints its CSV after every run (atomic rename), so a crash
  6 h into Phase 1 doesn't lose prior results.
- Phase 5 (PGLib-UC) is enabled by default. To run it:

  ```bash
  python uc_experiment_v3/fetch_pglib_uc.py                 # clones ./pglib-uc
  python uc_experiment_v3/run_v3_all.py --phase 5
  ```

  By default Phase 5 truncates each PGLib instance to 24 hours
  (`--horizon 24`) and skips `large` instances (>100 thermals). Renewables
  are not netted out of demand by default — see the docstring in
  `pglib_uc_adapter.py` for the reasoning. Pass `--net-renewables` to opt
  in to the renewable-floor subtraction (closer to PGLib's intended obj,
  but can over-constrain feasibility).
