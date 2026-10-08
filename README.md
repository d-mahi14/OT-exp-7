# Multiobjective Optimization Lab (Assignment 7)

## Environment
Run all commands with Python 3.12 from the repository root:
```powershell
py -3.12 -m pytest -q
```

---

## 1. Validation & Smoke Tests
To validate the 37 algorithm interfaces, metrics, determinism, and full unit test suite:
```powershell
py -3.12 -m experiments.smoke --with-tests
```
This runs:
- Metric calculation hand-checks (HV, IGD, Spacing)
- Mini experiment across evolutionary, sweep, decision (TOPSIS), single-solution (Value Function), and gradient (MGDA) methods
- Task C sensitivity parameter checks (PESA-II `n_divs`, MOBA `alpha`, Weight Metric `p`)
- Determinism check across identical seeds
- Convergence trajectory and specialized plot generation
- Full unit test suite (108 tests passing)

---

## 2. Running Experiments

The experiment runner is located at `experiments/run_all.py`.

### Task A (All 37 Algorithms across test problems)
```powershell
py -3.12 -m experiments.run_all --task A --resume
```

### Task C (Deep-Dive Sensitivity for Roll No. 17)
Roll No. 17 assigned algorithms:
- **Algorithm #17 (PESA-II)** on P3 and P5 with `n_divs` $\in [4, 8, 16]$
- **Algorithm #29 (MOBA)** on P3 and P4 with `alpha` $\in [0.7, 0.9, 0.99]$
- **Algorithm #4 (Weight Metric)** on P1 and P3 with `p` $\in [1, 2, \infty]$

```powershell
py -3.12 -m experiments.run_all --task C --resume
```

### Task D (Head-to-Head Comparisons)
- **Task D1 (Continuous Bi-Objective: P3 & P5)**:
  ```powershell
  py -3.12 -m experiments.run_all --task D1 --resume
  ```
- **Task D2 (Many-Objective: P10 with $M=5$)**:
  Compares NSGA-II (#14), NSGA-III (#15), MOEA/D (#18), RVEA (#31), and HypE (#34):
  ```powershell
  py -3.12 -m experiments.run_all --task D2 --resume
  ```
- **Task D3 (Discrete TSP: P8)**:
  Compares MOEA/D-ACO (#20), MOACO (#25), NSGA-II (#14), and MOMA (#30):
  ```powershell
  py -3.12 -m experiments.run_all --task D3 --resume
  ```

### Full Benchmark Suite (All Tasks)
```powershell
py -3.12 -m experiments.run_all --task all --resume
```

---

## 3. Useful Flags & Options
- `--resume`: Skip runs already recorded with `status=ok` in the raw CSV. Failed runs are automatically re-attempted.
- `--budget <N>`: Override evaluation budget (default: 10,000 evaluations).
- `--n-seeds <N>`: Run a subset of seeds (default: 10 seeds, 101–110).
- `--alg <ID>`: Run only a specific algorithm (1–37). Example:
  ```powershell
  py -3.12 -m experiments.run_all --task A --alg 14
  ```
- `--plots-only`: Skip execution and regenerate all plots and summaries from existing CSVs:
  ```powershell
  py -3.12 -m experiments.run_all --plots-only
  ```
- `--inspect`: Print the inspected `run()` signatures of all 37 algorithms:
  ```powershell
  py -3.12 -m experiments.run_all --inspect
  ```

---

## 4. Output Directories & Artifacts

All outputs are saved cleanly under `results/`:
- **`results/raw/`**:
  - `runs_full.csv`: Complete run-by-run log containing task, algorithm, problem, seed, parameters, actual evaluations used, runtime, HV, IGD, Spacing, and status.
  - `summary_full.csv`: Aggregated performance table reporting mean ± std over independent seeds.
  - `convergence_full.csv`: Checkpoint evaluations along the optimization trajectories.
  - `config_snapshot_full.json`: Reproducibility snapshot recording environment versions, seeds, and configurations.
  - `seeds.json`: Fixed seeds array `[101..110]`.
- **`results/fronts/`**:
  - Non-dominated Pareto front coordinates for each run saved as `a<ID>_<Problem>_s<Seed>.csv`.
- **`results/plots/`**:
  - Pareto front plots (`full_fronts_<Problem>_s<Seed>.png`)
  - Metric bar charts with error bars (`full_hv_bars_<Problem>.png`, `full_spacing_bars_<Problem>.png`, etc.)
  - Convergence trajectory curves (`full_convergence_<Problem>.png`)
  - Task C parameter sensitivity charts (`full_task_c_alg<ID>_<Problem>_sensitivity.png`)
  - Task D2 many-objective comparison (`full_task_d2_p10_m5.png`)
  - Task D3 discrete comparison (`full_task_d3_p8.png`)
  - MCDM ranking and weight sensitivity (`full_mcdm_p12.png`)
  - Gradient multi-task dynamics (`full_gradient_dynamics_p13.png`)
