"""
Algorithm 22: MOPSO (Multi-Objective Particle Swarm Optimization)
Proposed by: Coello Coello & Lechuga (2002)
Family: Swarm (Part 4, slides 13-20)

Core idea:
Extends PSO with an external archive of non-dominated solutions:
Each particle maintains a personal best (pbest) updated by Pareto dominance.
The global guide (gbest) for each particle is chosen from the external archive
based on crowding distance to favor less crowded regions:

    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    x_new = x + v_new

Archive maintenance:
- New non-dominated positions enter the archive; dominated members are removed.
- When archive size exceeds capacity, prune members with smallest crowding distance.

Default parameters:
- w = 0.7 (inertia weight)
- c1 = 1.5, c2 = 1.5 (cognitive and social constants)
"""

import numpy as np
import moo_utils as U


def select_gbest_from_archive(archive_X, archive_F, archive_CV, rng):
    """
    Select gbest leader from archive favoring solutions with larger crowding distance.
    """
    if len(archive_X) == 1:
        return archive_X[0]

    cd = U.crowding_distance(archive_F)
    # Finite crowding weights (replace inf with 2 * max finite)
    finite_mask = ~np.isinf(cd)
    max_finite = np.max(cd[finite_mask]) if np.any(finite_mask) else 1.0
    weights = np.where(np.isinf(cd), 2.0 * max_finite, cd)
    weights = np.maximum(weights, 1e-6)
    probs = weights / np.sum(weights)

    idx = rng.choice(len(archive_X), p=probs)
    return archive_X[idx]


def update_archive(archive_X, archive_F, archive_CV, new_X, new_F, new_CV, max_size):
    """
    Update archive with non-dominated solutions; prune by crowding distance if needed.
    """
    if len(archive_X) == 0:
        comb_X = new_X
        comb_F = new_F
        comb_CV = new_CV
    else:
        comb_X = np.vstack([archive_X, new_X])
        comb_F = np.vstack([archive_F, new_F])
        comb_CV = np.concatenate([archive_CV, new_CV])

    nd_idx = U.pareto_front(comb_F, comb_CV)
    arch_X = comb_X[nd_idx]
    arch_F = comb_F[nd_idx]
    arch_CV = comb_CV[nd_idx]

    while len(arch_X) > max_size:
        cd = U.crowding_distance(arch_F)
        # Prune interior solution with smallest crowding distance
        worst_idx = np.argmin(cd)
        arch_X = np.delete(arch_X, worst_idx, axis=0)
        arch_F = np.delete(arch_F, worst_idx, axis=0)
        arch_CV = np.delete(arch_CV, worst_idx, axis=0)

    return arch_X, arch_F, arch_CV


def initialize(problem, swarm_size=50, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(swarm_size, rng)
    V = np.zeros_like(X)
    F, CV = problem.evaluate(X)
    pbest_X = X.copy()
    pbest_F = F.copy()
    pbest_CV = CV.copy()

    # Initial archive
    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), np.empty(0),
        X, F, CV, archive_size
    )
    return X, V, F, CV, pbest_X, pbest_F, pbest_CV, arch_X, arch_F, arch_CV


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, V, F, CV, pbest_X, pbest_F, pbest_CV, arch_X, arch_F, arch_CV,
         w=0.7, c1=1.5, c2=1.5, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N, d = X.shape

    # Update each particle
    for i in range(N):
        gbest = select_gbest_from_archive(arch_X, arch_F, arch_CV, rng)
        r1 = rng.uniform(0.0, 1.0, size=d)
        r2 = rng.uniform(0.0, 1.0, size=d)

        V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest - X[i])
        X[i] = np.clip(X[i] + V[i], problem.lower, problem.upper)
        if problem.var_type == "int":
            X[i] = np.round(X[i])

    F, CV = evaluate(problem, X)

    # Update personal bests
    for i in range(N):
        if U.constrained_dominates(F[i], CV[i], pbest_F[i], pbest_CV[i]):
            pbest_X[i] = X[i].copy()
            pbest_F[i] = F[i].copy()
            pbest_CV[i] = CV[i]
        elif not U.constrained_dominates(pbest_F[i], pbest_CV[i], F[i], CV[i]):
            if rng.random() < 0.5:
                pbest_X[i] = X[i].copy()
                pbest_F[i] = F[i].copy()
                pbest_CV[i] = CV[i]

    # Update archive
    arch_X, arch_F, arch_CV = update_archive(
        arch_X, arch_F, arch_CV, X, F, CV, archive_size
    )

    return X, V, F, CV, pbest_X, pbest_F, pbest_CV, arch_X, arch_F, arch_CV


def run(problem, swarm_size=50, archive_size=100, n_gen=50, w=0.7, c1=1.5, c2=1.5, seed=42):
    rng = np.random.default_rng(seed)
    (X, V, F, CV, pbest_X, pbest_F, pbest_CV,
     arch_X, arch_F, arch_CV) = initialize(problem, swarm_size=swarm_size, archive_size=archive_size, rng=rng)

    for gen in range(n_gen):
        (X, V, F, CV, pbest_X, pbest_F, pbest_CV,
         arch_X, arch_F, arch_CV) = step(
            problem, X, V, F, CV, pbest_X, pbest_F, pbest_CV, arch_X, arch_F, arch_CV,
            w=w, c1=c1, c2=c2, archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
