"""Full suite.  py -3.12 -m experiments.run_all --task all --resume
   Interfaces: py -3.12 -m experiments.run_all --inspect"""
import argparse, inspect
from experiments import config as C
from experiments import runner as R
from experiments import plots as P

TASKS = ["A", "C", "D1", "D2", "D3"]


def inspect_all():
    for a, (stem, name, kind, tests) in C.CATALOGUE.items():
        try:
            sig = inspect.signature(R.load_algorithm(a).run)
        except Exception as e:
            sig = f"IMPORT/INSPECT ERROR {e!r}"
        print(f"{a:>2} {name:<20} [{kind}] run{sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=TASKS + ["all"])
    ap.add_argument("--budget", type=int, default=C.BUDGET)
    ap.add_argument("--n-seeds", type=int, default=len(C.SEEDS))
    ap.add_argument("--resume", action="store_true", help="skip runs already ok in the CSV")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--alg", type=int, default=None, help="Filter to a single algorithm id (1-37)")
    a = ap.parse_args()
    if a.inspect:
        return inspect_all()
    tag = "full"
    seeds = C.SEEDS[:a.n_seeds]
    R.write_snapshot(tag, {"tasks": a.task, "n_seeds": a.n_seeds, "budget": a.budget, "alg": a.alg})
    if not a.plots_only:
        for t in (TASKS if a.task == "all" else [a.task]):
            jobs = C.jobs_for(t, seeds)
            if a.alg is not None:
                jobs = [j for j in jobs if j["alg"] == a.alg]
            print(f"=== task {t}: {len(jobs)} runs ===")
            if jobs:
                R.run_batch(jobs, tag, budget=a.budget, resume=a.resume)
    for o in P.make_all(tag):
        print("wrote", o)


if __name__ == "__main__":
    main()