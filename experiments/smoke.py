"""Small validation run. Usage (repo root):  py -3.12 -m experiments.smoke [--with-tests]"""
import argparse, shutil, subprocess, sys
import numpy as np
from experiments import config as C
from experiments import runner as R
from experiments import plots as P
import moo_utils as U

TAG = "smoke"
SEEDS = [101, 102]
BUDGET, POP = 1000, 40
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-tests", action="store_true")
    args = ap.parse_args()

    for p in (C.RAW / f"runs_{TAG}.csv", C.RAW / f"convergence_{TAG}.csv", C.RAW / f"summary_{TAG}.csv"):
        p.unlink(missing_ok=True)
    shutil.rmtree(C.FRONTS / TAG, ignore_errors=True)
    for f in C.PLOTS.glob(f"{TAG}_*.png"):
        f.unlink()

    print("== 1. metric hand-checks ==")
    two = np.array([[1.0, 2.0], [2.0, 1.0]])
    check("HV of {(1,2),(2,1)} with ref (3,3) == 3", abs(U.hv_2d(two, np.array([3.0, 3.0])) - 3.0) < 1e-9)
    even = np.array([[0, 3], [1, 2], [2, 1], [3, 0]], float)
    check("Spacing of evenly spaced front == 0", abs(U.spacing(even)) < 1e-9)
    check("IGD(front, front) == 0", abs(U.igd(even, even)) < 1e-9)

    print("== 2. mini experiment (budget 1000, pop 40, seeds 101,102) ==")
    jobs = []
    # Test evolutionary, sweep, and all custom adapter types:
    for a, pids in ((14, ["P3", "P5"]), (16, ["P5"]), (1, ["P1"]), (5, ["P4"]), (8, ["P12"]), (36, ["P13"])):
        jobs += [dict(task="SMOKE", alg=a, pid=p, seed=s, params={}) for p in pids for s in SEEDS]
    # Also test Task C deep-dive parameters (17 n_divs, 29 alpha, 4 p)
    jobs += [
        dict(task="SMOKE_C", alg=17, pid="P3", seed=101, params={"n_divs": 4}),
        dict(task="SMOKE_C", alg=29, pid="P3", seed=101, params={"alpha": 0.7}),
        dict(task="SMOKE_C", alg=4, pid="P1", seed=101, params={"p": 1}),
    ]
    rows = R.run_batch(jobs, TAG, budget=BUDGET, pop_override=POP)
    R.write_snapshot(TAG, {"smoke": True})
    ok_rows = [r for r in rows if r["status"] == "ok"]
    for r in rows:
        if r["status"] != "ok":
            print(f"      NOTE {r['alg_name']} {r['problem']} s{r['seed']}: {r['status']} {r['error']}")

    print("== 3. output checks ==")
    check("runs CSV written", (C.RAW / f"runs_{TAG}.csv").exists())
    check("all runs ok", len(ok_rows) == len(jobs), f"{len(ok_rows)}/{len(jobs)} ok")
    check("seeds recorded in CSV", {str(s) for s in SEEDS}.issubset({str(r["seed"]) for r in rows}))
    check("HV finite and >= 0", all(np.isfinite(float(r["hv"])) and float(r["hv"]) >= 0 for r in ok_rows))
    check("Spacing finite and >= 0", all(np.isfinite(float(r["spacing"])) and float(r["spacing"]) >= 0
                                         for r in ok_rows))
    check("evals counted correctly", all(int(r["evals_used"]) >= 0 for r in ok_rows) and
                                    all(int(r["evals_used"]) > 0 for r in ok_rows if int(r["alg_id"]) not in (8, 9)))
    check("front files saved", len(list((C.FRONTS / TAG).glob("*.csv"))) >= len(ok_rows) > 0)
    check("config snapshot + seeds.json written",
          (C.RAW / f"config_snapshot_{TAG}.json").exists() and (C.RAW / "seeds.json").exists())

    print("== 4. determinism (same seed twice) ==")
    j = dict(task="SMOKE", alg=14, pid="P3", seed=101, params={})
    _, F1 = R.run_one(j, BUDGET, POP, TAG, save=False)
    _, F2 = R.run_one(j, BUDGET, POP, TAG, save=False)
    check("NSGA-II P3 seed 101 reproduces identical front",
          F1 is not None and F2 is not None and F1.shape == F2.shape and np.allclose(F1, F2))

    print("== 5. convergence + plots ==")
    R.run_convergence(dict(task="SMOKE", alg=14, pid="P3", seed=101, params={}), [200, 500, 1000], TAG)
    outs = P.make_all(TAG)
    for o in outs:
        print("      wrote", o)
    check("plots generated", len(outs) > 0 and all(o.exists() for o in outs))
    check("summary CSV written", (C.RAW / f"summary_{TAG}.csv").exists())

    if args.with_tests:
        print("== 6. existing test suite ==")
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=C.ROOT).returncode
        check("pytest still passes", rc == 0)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()