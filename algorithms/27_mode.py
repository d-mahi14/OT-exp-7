"""
Algorithm 27: MODE (Multi-Objective Differential Evolution)
Proposed by: Xue, Sanderson & Graves (2003)
Family: Evolutionary (Part 4, slides 49-55)

Core idea:
Applies Differential Evolution (DE) mutation and binomial crossover to generate
a trial vector for each target individual:

    V = X_r1 + F * (X_r2 - X_r3)
    U = crossover(X_target, V, CR)

Selection rule:
- Trial U replaces target X_target ONLY IF U Pareto-dominates it (with constraints).
- Otherwise, target survives unchanged and trial U enriches the external archive
  if it is non-dominated.
- An external archive preserves all non-dominated solutions discovered.

Default parameters:
- F = 0.5 (differential weight)
- CR = 0.9 (crossover probability)
"""

import numpy as np
import moo_utils as U
import operators as ops


def update_archive(archive_X, archive_F, archive_CV, new_X, new_F, new_CV, max_size=100):
    if len(archive_X) == 0:
        comb_X = np.atleast_2d(new_X)
        comb_F = np.atleast_2d(new_F)
        comb_CV = np.atleast_1d(new_CV)
    else:
        comb_X = np.vstack([archive_X, np.atleast_2d(new_X)])
        comb_F = np.vstack([archive_F, np.atleast_2d(new_F)])
        comb_CV = np.concatenate([archive_CV, np.atleast_1d(new_CV)])

    nd_idx = U.pareto_front(comb_F, comb_CV)
    arch_X = comb_X[nd_idx]
    arch_F = comb_F[nd_idx]
    arch_CV = comb_CV[nd_idx]

    while len(arch_X) > max_size:
        cd = U.crowding_distance(arch_F)
        worst_idx = np.argmin(cd)
        arch_X = np.delete(arch_X, worst_idx, axis=0)
        arch_F = np.delete(arch_F, worst_idx, axis=0)
        arch_CV = np.delete(arch_CV, worst_idx, axis=0)

    return arch_X, arch_F, arch_CV


def initialize(problem, pop_size=50, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pop_size, rng)
    F, CV = problem.evaluate(X)
    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), np.empty(0),
        X, F, CV, archive_size
    )
    return X, F, CV, arch_X, arch_F, arch_CV


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, arch_X, arch_F, arch_CV, F_scale=0.5, CR=0.9, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    for i in range(N):
        # Select 3 distinct individuals != i
        candidates = [idx for idx in range(N) if idx != i]
        r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

        # 1. DE mutation
        v = ops.de_mutation(X[r1], X[r2], X[r3], F=F_scale)

        # 2. Binomial crossover
        u = ops.de_binomial_crossover(X[i], v, CR=CR, rng=rng)
        u = np.clip(u, problem.lower, problem.upper)
        if problem.var_type == "int":
            u = np.round(u)

        fu, cvu = problem.evaluate(u)
        fu = fu[0] if fu.ndim > 1 else fu
        cvu = cvu[0] if np.ndim(cvu) > 0 else cvu

        # 3. Selection: trial replaces target ONLY IF trial dominates target
        if U.constrained_dominates(fu, cvu, F[i], CV[i]):
            X[i] = u.copy()
            F[i] = fu.copy()
            CV[i] = cvu
        # Enriches archive with the trial
        arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, u, fu, cvu, archive_size)

    return X, F, CV, arch_X, arch_F, arch_CV


def run(problem, pop_size=50, archive_size=100, n_gen=50, F_scale=0.5, CR=0.9, seed=42):
    rng = np.random.default_rng(seed)
    X, F, CV, arch_X, arch_F, arch_CV = initialize(
        problem, pop_size=pop_size, archive_size=archive_size, rng=rng
    )

    for gen in range(n_gen):
        X, F, CV, arch_X, arch_F, arch_CV = step(
            problem, X, F, CV, arch_X, arch_F, arch_CV,
            F_scale=F_scale, CR=CR, archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
