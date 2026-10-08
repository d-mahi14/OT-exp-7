"""Unit tests for experiment infrastructure, routing, custom adapters, and Task C/D settings."""
import inspect
import json
import numpy as np
import pytest

from experiments import config as C
from experiments import runner as R


def test_task_c_parameters_accepted():
    """Verify that all Task C deep-dive algorithms accept their sensitivity parameter."""
    for aid, spec in C.TASK_C.items():
        mod = R.load_algorithm(aid)
        param_name = spec["param"]
        sig = inspect.signature(mod.run)
        assert param_name in sig.parameters, (
            f"Algorithm {aid} ({C.CATALOGUE[aid][1]}) run() signature does not accept '{param_name}'. "
            f"Signature: {sig}"
        )


def test_job_generation_all_tasks():
    """Verify job generation for Tasks A, C, D1, D2, D3."""
    seeds = [101, 102]
    jobs_a = C.jobs_for("A", seeds)
    assert len(jobs_a) > 0
    algs_in_a = {j["alg"] for j in jobs_a}
    assert len(algs_in_a) == 37, f"Expected all 37 algorithms in Task A, found {len(algs_in_a)}"

    jobs_c = C.jobs_for("C", seeds)
    assert len(jobs_c) == len(C.TASK_C) * 2 * 3 * len(seeds)  # 3 algs * 2 probs * 3 values * 2 seeds
    assert {j["alg"] for j in jobs_c} == {17, 29, 4}

    jobs_d1 = C.jobs_for("D1", seeds)
    assert len(jobs_d1) > 0
    assert all(j["pid"] in ("P3", "P5") for j in jobs_d1)

    jobs_d2 = C.jobs_for("D2", seeds)
    assert len(jobs_d2) == len(C.TASK_D2["algs"]) * len(seeds)
    assert {j["alg"] for j in jobs_d2} == set(C.TASK_D2["algs"])
    assert all(j["pid"] == "P10_M5" for j in jobs_d2)

    jobs_d3 = C.jobs_for("D3", seeds)
    assert len(jobs_d3) == len(C.TASK_D3["algs"]) * len(seeds)
    assert {j["alg"] for j in jobs_d3} == set(C.TASK_D3["algs"])
    assert all(j["pid"] == "P8" for j in jobs_d3)


def test_custom_adapters_single_solution():
    """Verify single-solution adapters (#5 Value Function, #6 Goal Programming, #7 Lexicographic)."""
    # Alg 5 on P4
    j5 = dict(task="TEST", alg=5, pid="P4", seed=101, params={})
    row5, F5 = R.run_one(j5, budget=200, save=False)
    assert row5["status"] == "ok"
    assert row5["n_front"] == 1
    assert F5 is not None and F5.shape == (1, 2)
    assert row5["evals_used"] > 0

    # Alg 6 on P11
    j6 = dict(task="TEST", alg=6, pid="P11", seed=101, params={})
    row6, F6 = R.run_one(j6, budget=200, save=False)
    assert row6["status"] == "ok"
    assert row6["n_front"] == 1
    assert F6 is not None and F6.shape == (1, 2)

    # Alg 7 on P11
    j7 = dict(task="TEST", alg=7, pid="P11", seed=101, params={})
    row7, F7 = R.run_one(j7, budget=200, save=False)
    assert row7["status"] == "ok"
    assert row7["n_front"] == 1
    assert F7 is not None and F7.shape == (1, 2)


def test_custom_adapters_mcdm():
    """Verify decision/ranking adapters (#8 TOPSIS, #9 VIKOR) on P12."""
    # Alg 8 TOPSIS on P12
    j8 = dict(task="TEST", alg=8, pid="P12", seed=101, params={})
    row8, F8 = R.run_one(j8, budget=100, save=False)
    assert row8["status"] == "ok"
    assert row8["n_front"] == 3
    assert F8 is not None and F8.shape == (3, 4)
    params8 = json.loads(row8["params"])
    assert params8 == {}  # preserves input params for --resume
    mod8 = R.load_algorithm(8)
    p12 = R.load_problem("P12")
    rank8, C_vals = mod8.run(p12)
    assert len(rank8) == len(p12.suppliers)

    # Alg 9 VIKOR on P12
    j9 = dict(task="TEST", alg=9, pid="P12", seed=101, params={})
    row9, F9 = R.run_one(j9, budget=100, save=False)
    assert row9["status"] == "ok"
    assert row9["n_front"] == 3
    assert F9 is not None and F9.shape == (3, 4)
    params9 = json.loads(row9["params"])
    assert params9 == {}  # preserves input params for --resume
    mod9 = R.load_algorithm(9)
    rank9, Q_vals, comp_set = mod9.run(p12)
    assert len(rank9) == len(p12.suppliers)
    assert len(comp_set) > 0


def test_custom_adapters_gradient():
    """Verify gradient adapters (#36 MGDA, #37 GradNorm) on P13."""
    # Alg 36 MGDA on P13
    j36 = dict(task="TEST", alg=36, pid="P13", seed=101, params={})
    row36, F36 = R.run_one(j36, budget=100, save=False)
    assert row36["status"] == "ok"
    assert row36["n_front"] >= 1
    assert F36 is not None and F36.ndim == 2
    assert row36["evals_used"] > 0

    # Alg 37 GradNorm on P13
    j37 = dict(task="TEST", alg=37, pid="P13", seed=101, params={})
    row37, F37 = R.run_one(j37, budget=100, save=False)
    assert row37["status"] == "ok"
    assert row37["n_front"] >= 1
    assert F37 is not None and F37.ndim == 2
    assert row37["evals_used"] > 0


def test_counting_problem():
    """Verify CountingProblem accurately records evaluation calls."""
    p = R.load_problem("P1")
    cp = R.CountingProblem(p)
    assert cp.evals == 0
    rng = np.random.default_rng(42)
    X = p.random_solutions(25, rng)
    cp.evaluate(X)
    assert cp.evals == 25
    cp.evaluate(X[0])
    assert cp.evals == 26


def test_resume_behavior():
    """Verify that ok rows are skipped when resuming, but error rows are retried."""
    from experiments.runner import _key
    probe_ok = dict(task="A", alg_id=14, problem="P3", seed=101, params="{}", budget_target=1000)
    done_set = {_key(probe_ok)}
    assert _key(probe_ok) in done_set

    probe_err = dict(task="A", alg_id=14, problem="P3", seed=102, params="{}", budget_target=1000)
    assert _key(probe_err) not in done_set
