"""Fixed-seed experiment runner: one raw CSV row per (task, algorithm, problem, seed, params)."""
import csv, importlib, inspect, json, platform, subprocess, sys, time, traceback
from datetime import datetime
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import moo_utils as U
import problems as PM
from experiments import config as C

FIELDS = ["task", "alg_id", "alg_name", "problem", "seed", "params", "budget_target", "pop_size",
          "n_gen", "evals_used", "n_front", "hv", "igd", "spacing", "runtime_s", "status", "error"]
CONV_FIELDS = ["task", "alg_id", "alg_name", "problem", "seed", "params", "budget", "evals_used",
               "hv", "igd", "spacing", "status", "error"]


# ---------- problem / algorithm loading ----------
def parse_pid(pid):
    return ("P10", int(pid.split("M")[1])) if pid.startswith("P10_M") else (pid, None)


def _instantiate(obj, M):
    if not isinstance(obj, type) and hasattr(obj, "evaluate"):
        return obj
    attempts = [lambda: obj()] if M is None else [lambda: obj(M=M), lambda: obj(M), lambda: obj(n_obj=M)]
    err = None
    for f in attempts:
        try:
            return f()
        except Exception as e:
            err = e
    raise err


def load_problem(pid):
    base, M = parse_pid(pid)
    if base.upper() in ("P13", "P13A"):
        return PM.ToyTwoTask()
    if base.upper() == "P13B":
        return PM.MultiTaskNet()
    for name in ("get_problem", "make_problem"):
        f = getattr(PM, name, None)
        if callable(f):
            for call in (lambda: f(base, M=M) if M else f(base), lambda: f(pid), lambda: f(base)):
                try:
                    return call()
                except Exception:
                    pass
    for name in ("PROBLEMS", "PROBLEM_BANK", "REGISTRY", "problems"):
        reg = getattr(PM, name, None)
        if isinstance(reg, dict):
            for key in (pid, base, base.lower(), int(base[1:])):
                if key in reg:
                    return _instantiate(reg[key], M)
    for n in dir(PM):
        if not n.startswith("_") and (n.upper() == base or n.upper().startswith(base + "_")):
            return _instantiate(getattr(PM, n), M)
    raise LookupError(f"Cannot locate problem {pid} in problems.py. Public names: "
                      f"{[n for n in dir(PM) if not n.startswith('_')]}")


def load_algorithm(alg_id):
    return importlib.import_module(f"algorithms.{C.CATALOGUE[alg_id][0]}")


def get_true_front(problem, n=C.N_TRUE):
    for name in ("true_front", "true_pareto_front", "pareto_front", "reference_front"):
        a = getattr(problem, name, None)
        try:
            v = a(n) if callable(a) else a
            if v is not None:
                v = np.asarray(v, float)
                if v.ndim == 2:
                    return v
        except Exception:
            pass
    return None


class CountingProblem:
    """Delegates everything to the real problem and counts rows passed to evaluate()."""
    def __init__(self, p):
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "evals", 0)

    def __getattr__(self, name):
        return getattr(self._p, name)

    def evaluate(self, X, *a, **k):
        object.__setattr__(self, "evals", self.evals + len(np.atleast_2d(np.asarray(X))))
        return self._p.evaluate(X, *a, **k)


def n_objectives(problem):
    n = getattr(problem, "n_obj", None)
    if n:
        return int(n)
    F, _ = problem.evaluate(problem.random_solutions(1, np.random.default_rng(0)))
    return int(np.asarray(F).shape[1])


def extract_front(result, n_obj):
    if isinstance(result, dict):
        for k in ("F_front", "F", "front", "objectives"):
            if k in result:
                return np.atleast_2d(np.asarray(result[k], float))
        raise ValueError(f"dict result without a front key: {list(result)}")
    items = list(reversed(result)) if isinstance(result, (tuple, list)) else [result]
    for it in items:
        try:
            a = np.asarray(it, dtype=float)
        except Exception:
            continue
        if a.ndim == 2 and a.shape[1] == n_obj:
            return a
    raise ValueError(f"no (n, {n_obj}) array found in algorithm result of type {type(result)}")


def compute_metrics(problem, F):
    out = U.front_metrics(problem, F, n_true=C.N_TRUE, hv_samples=C.HV_SAMPLES, seed=0)
    if not isinstance(out, dict):
        raise TypeError(f"front_metrics returned {type(out)}, expected dict: {out!r}")
    low = {str(k).lower(): v for k, v in out.items()}
    pick = lambda *ks: next((low[k] for k in ks if k in low), None)
    hv, igd, sp = pick("hv", "hypervolume"), pick("igd"), pick("spacing")
    if hv is None or sp is None:
        raise KeyError(f"front_metrics keys not recognised: {list(out)}")
    return float(hv), (np.nan if igd is None else float(igd)), float(sp)


# ---------- single run ----------
def build_kwargs(fn, cand):
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(cand), []
    kw = {k: v for k, v in cand.items() if k in sig.parameters}
    return kw, [k for k in cand if k not in sig.parameters]


def run_one(job, budget=C.BUDGET, pop_override=None, tag="adhoc", save=True):
    a, pid, seed = job["alg"], job["pid"], job["seed"]
    stem, name, kind, _ = C.CATALOGUE[a]
    pop = pop_override or C.POP_BY_PROBLEM.get(pid, C.DEFAULT_POP)
    n_gen = max(1, budget // pop - 1)
    row = dict(task=job["task"], alg_id=a, alg_name=name, problem=pid, seed=seed,
               params=json.dumps(job["params"], default=str), budget_target=budget,
               pop_size=pop if kind == "evo" else "", n_gen=n_gen if kind == "evo" else "",
               evals_used="", n_front="", hv="", igd="", spacing="", runtime_s="",
               status="ok", error="")
    try:
        problem = load_problem(pid)
        mod = load_algorithm(a)
        fn = getattr(mod, "run")
        t0 = time.perf_counter()

        for p in job["params"]:
            if p not in inspect.signature(fn).parameters:
                row["status"] = "param_not_accepted"
                row["error"] = f"run() of {stem} has no parameter '{p}'"
                return row, None

        if a in (8, 9):
            # MCDM / Ranking methods on P12 (no continuous evaluation budget)
            if a == 8:  # TOPSIS
                rankings, C_vals = mod.run(problem)
                w_shift = np.array([0.2, 0.4, 0.2, 0.2])
                rankings_s, C_vals_s = mod.run(problem, weights=w_shift)
                top3 = [problem.suppliers[i] for i in rankings[:3]]
                F = problem.matrix[rankings[:3]]
                row.update(evals_used=0, n_front=len(top3),
                           hv=round(float(C_vals[rankings[0]]), 4),
                           spacing=round(float(np.std(C_vals)), 4), igd="")
            else:  # VIKOR
                rankings, Q_vals, comp_set = mod.run(problem)
                w_shift = np.array([0.2, 0.4, 0.2, 0.2])
                rankings_s, Q_vals_s, _ = mod.run(problem, weights=w_shift)
                top3 = [problem.suppliers[i] for i in rankings[:3]]
                F = problem.matrix[rankings[:3]]
                row.update(evals_used=0, n_front=len(top3),
                           hv=round(float(1.0 - Q_vals[rankings[0]]), 4),
                           spacing=round(float(np.std(Q_vals)), 4), igd="")

        elif a in (36, 37):
            # Gradient-based multi-task methods on P13 (ToyTwoTask)
            cp = CountingProblem(problem)
            steps = min(100, max(10, budget // 10))
            if a == 36:  # MGDA
                cand = {"max_steps": steps, "eta": 0.1}
                cand.update(job["params"])
                kw, _ = build_kwargs(fn, cand)
                traj_x, traj_f = fn(cp, **kw)
                F_all = np.atleast_2d(traj_f)
                nd = U.pareto_front(F_all)
                F = F_all[nd]
                hv, igd, sp = compute_metrics(problem, F)
                row.update(evals_used=cp.evals, n_front=len(F), hv=hv, spacing=sp,
                           igd="" if np.isnan(igd) else igd)
            else:  # GradNorm
                cand = {"n_steps": steps, "seed": seed, "lr_theta": 0.01, "lr_w": 0.025, "alpha_asym": 0.5}
                cand.update(job["params"])
                kw, _ = build_kwargs(fn, cand)
                theta, losses, weights = fn(cp, **kw)
                F_all = np.atleast_2d(losses)
                nd = U.pareto_front(F_all)
                F = F_all[nd]
                hv, igd, sp = compute_metrics(problem, F)
                row.update(evals_used=cp.evals, n_front=len(F), hv=hv, spacing=sp,
                           igd="" if np.isnan(igd) else igd)

        elif a in (5, 6, 7):
            # Single-solution compromise / preference methods
            cp = CountingProblem(problem)
            cand = {"seed": seed, "n_candidates": min(budget, C.SWEEP_CANDIDATES)}
            cand.update(job["params"])
            kw, _ = build_kwargs(fn, cand)
            res = fn(cp, **kw)
            bx, bf = res
            F = np.atleast_2d(np.asarray(bf, dtype=float))
            hv, igd, sp = compute_metrics(problem, F)
            row.update(evals_used=cp.evals, n_front=1, hv=hv, spacing=sp,
                       igd="" if np.isnan(igd) else igd)

        else:
            # kind == "evo" or kind == "sweep"
            cp = CountingProblem(problem)
            cand = {"seed": seed}
            if kind == "evo":
                cand.update(pop_size=pop, n_gen=n_gen)
            else:
                cand.update(n_weights=C.SWEEP_N, n_candidates=C.SWEEP_CANDIDATES)
            cand.update(job["params"])
            kw, _ = build_kwargs(fn, cand)
            res = fn(cp, **kw)
            F = extract_front(res, n_objectives(problem))
            hv, igd, sp = compute_metrics(problem, F)
            row.update(evals_used=cp.evals, n_front=len(F), hv=hv, spacing=sp,
                       igd="" if np.isnan(igd) else igd)

        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        if save and F is not None and len(F) > 0:
            d = C.FRONTS / tag
            d.mkdir(parents=True, exist_ok=True)
            np.savetxt(d / f"a{a:02d}_{pid}_s{seed}.csv", F, delimiter=",")
        return row, F
    except Exception as e:
        row["status"] = "error"
        row["error"] = (repr(e) + " | " + traceback.format_exc().strip().splitlines()[-3].strip())[:400]
        return row, None


# ---------- batches, resume, snapshots ----------
def _input_params_key(r):
    """Extract canonical input parameter string, filtering legacy output metadata."""
    p_str = r.get("params", "") if isinstance(r, dict) else getattr(r, "params", "")
    if not p_str or p_str == "{}":
        return "{}"
    try:
        p_dict = json.loads(p_str)
        input_dict = {k: v for k, v in p_dict.items()
                      if k not in ("best_f", "top3", "top3_shifted", "rank1_flipped", "C_top",
                                   "Q_top", "compromise_set", "steps", "final_loss", "final_weights")}
        return json.dumps(input_dict, sort_keys=True)
    except Exception:
        return p_str


def _key(r):
    return (str(r["task"]), str(r["alg_id"]), str(r["problem"]), str(r["seed"]),
            _input_params_key(r), str(r.get("budget_target", C.BUDGET)))


def _append(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_batch(jobs, tag, budget=C.BUDGET, pop_override=None, resume=False, verbose=True):
    path = C.RAW / f"runs_{tag}.csv"
    done = set()
    if resume and path.exists():
        done = {_key(r) for r in read_csv(path) if r["status"] == "ok"}
    out = []
    for i, job in enumerate(jobs, 1):
        probe = dict(task=job["task"], alg_id=job["alg"], problem=job["pid"], seed=job["seed"],
                     params=json.dumps(job["params"], default=str), budget_target=budget)
        if _key(probe) in done:
            continue
        row, _ = run_one(job, budget, pop_override, tag)
        _append(path, FIELDS, [row])
        out.append(row)
        if verbose:
            print(f"[{i}/{len(jobs)}] {row['task']} {row['alg_name']:<20} {row['problem']:<7} "
                  f"seed={row['seed']} {row['status']} hv={row['hv']} t={row['runtime_s']}s", flush=True)
    return out


def run_convergence(job, checkpoints, tag):
    """HV/IGD along budget checkpoints. Uses a single continuous trajectory if supported, else checkpoints."""
    a, pid, seed = job["alg"], job["pid"], job["seed"]
    stem, name, kind, _ = C.CATALOGUE[a]
    problem = load_problem(pid)
    mod = load_algorithm(a)
    rows = []

    if kind == "evo" and hasattr(mod, "initialize") and hasattr(mod, "step"):
        try:
            pop = C.POP_BY_PROBLEM.get(pid, C.DEFAULT_POP)
            rng = np.random.default_rng(seed)
            cp = CountingProblem(problem)
            X = mod.initialize(cp, pop_size=pop, rng=rng)
            F, CV = mod.evaluate(cp, X) if hasattr(mod, "evaluate") else cp.evaluate(X)
            checkpoints_sorted = sorted(checkpoints)
            chk_idx = 0
            max_b = checkpoints_sorted[-1]
            while cp.evals < max_b and chk_idx < len(checkpoints_sorted):
                target_b = checkpoints_sorted[chk_idx]
                if cp.evals >= target_b:
                    nd = U.pareto_front(F, CV)
                    F_nd = F[nd]
                    hv, igd, sp = compute_metrics(problem, F_nd)
                    rows.append(dict(task=job["task"], alg_id=a, alg_name=name, problem=pid,
                                     seed=seed, params=json.dumps(job["params"], default=str),
                                     budget=target_b, evals_used=cp.evals, hv=hv,
                                     igd="" if np.isnan(igd) else igd, spacing=sp,
                                     status="ok", error=""))
                    chk_idx += 1
                X, F, CV = mod.step(cp, X, F, CV, rng=rng)

            while chk_idx < len(checkpoints_sorted):
                target_b = checkpoints_sorted[chk_idx]
                nd = U.pareto_front(F, CV)
                F_nd = F[nd]
                hv, igd, sp = compute_metrics(problem, F_nd)
                rows.append(dict(task=job["task"], alg_id=a, alg_name=name, problem=pid,
                                 seed=seed, params=json.dumps(job["params"], default=str),
                                 budget=target_b, evals_used=cp.evals, hv=hv,
                                 igd="" if np.isnan(igd) else igd, spacing=sp,
                                 status="ok", error=""))
                chk_idx += 1
        except Exception:
            rows = []

    if not rows:
        for b in checkpoints:
            r, _ = run_one(job, budget=b, tag=tag, save=False)
            rows.append({k: r.get(k, "") for k in CONV_FIELDS if k != "budget"} | {"budget": b})

    _append(C.RAW / f"convergence_{tag}.csv", CONV_FIELDS, rows)
    return rows


def write_snapshot(tag, extra=None):
    C.RAW.mkdir(parents=True, exist_ok=True)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=C.ROOT).stdout.strip()
    except Exception:
        git = ""
    snap = dict(tag=tag, timestamp=datetime.now().isoformat(timespec="seconds"),
                python=platform.python_version(), numpy=np.__version__, git_commit=git,
                seeds=C.SEEDS, budget=C.BUDGET, default_pop=C.DEFAULT_POP,
                pop_by_problem=C.POP_BY_PROBLEM, hv_samples=C.HV_SAMPLES, n_true=C.N_TRUE,
                sweep_n=C.SWEEP_N, sweep_candidates=C.SWEEP_CANDIDATES, extra=extra or {})
    (C.RAW / f"config_snapshot_{tag}.json").write_text(json.dumps(snap, indent=2))
    (C.RAW / "seeds.json").write_text(json.dumps(C.SEEDS))


def aggregate(tag):
    """Mean +/- std over seeds of the ok rows. Deduplicates by (task, alg_id, problem, seed, input_params),
    keeping the latest ok run for each seed. Writes summary_<tag>.csv; values are computed, never typed."""
    raw_path = C.RAW / f"runs_{tag}.csv"
    if not raw_path.exists():
        return []
    rows = [r for r in read_csv(raw_path) if r.get("status") == "ok"]

    # Deduplicate: for each (task, alg_id, problem, seed, input_params), keep the latest run
    latest_by_seed = {}
    for r in rows:
        k = (str(r["task"]), str(r["alg_id"]), str(r["problem"]), str(r["seed"]), _input_params_key(r))
        latest_by_seed[k] = r
    deduped_rows = list(latest_by_seed.values())

    # Group by (task, alg_id, alg_name, problem, canonical_input_params)
    groups = {}
    for r in deduped_rows:
        gkey = (str(r["task"]), str(r["alg_id"]), str(r["alg_name"]), str(r["problem"]), _input_params_key(r))
        groups.setdefault(gkey, []).append(r)

    f = lambda rs, k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rs])
    out = []
    for (t, aid, nm, pid, pr), rs in groups.items():
        d = dict(task=t, alg_id=aid, alg_name=nm, problem=pid, params=pr, n_runs=len(rs))
        for k in ("hv", "igd", "spacing", "runtime_s", "evals_used", "n_front"):
            v = f(rs, k)
            ok = ~np.isnan(v)
            d[k + "_mean"] = float(v[ok].mean()) if ok.any() else ""
            d[k + "_std"] = float(v[ok].std(ddof=1)) if ok.sum() > 1 else ""
        out.append(d)

    if out:
        p = C.RAW / f"summary_{tag}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return out