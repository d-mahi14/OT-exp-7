"""
Algorithm 23: MOGWO (Multi-Objective Grey Wolf Optimizer)
Proposed by: Mirjalili, Saremi, Mirjalili & Coelho (2016)
Family: Swarm (Part 4, slides 21-27)

Core idea:
Extends Grey Wolf Optimizer with an external archive of non-dominated solutions.
The three pack leaders (alpha, beta, delta) are selected each iteration from the archive
by crowding distance (favoring the least-crowded trade-off points).
Every omega wolf moves toward the leaders:

    X_new = (X_alpha + X_beta + X_delta) / 3

or full GWO encircling:
    D_alpha = |C_1 * X_alpha - X|,  X_1 = X_alpha - A_1 * D_alpha
    D_beta  = |C_2 * X_beta  - X|,  X_2 = X_beta  - A_2 * D_beta
    D_delta = |C_3 * X_delta - X|,  X_3 = X_delta - A_3 * D_delta
    X_new = (X_1 + X_2 + X_3) / 3

Archive maintenance:
- Pruned by crowding distance when size exceeds capacity.
"""

import numpy as np
import moo_utils as U


def select_leaders_from_archive(archive_X, archive_F, rng):
    """
    Select alpha, beta, delta leaders from archive by crowding distance (least crowded).
    """
    n_arch = len(archive_X)
    if n_arch == 0:
        return None, None, None
    if n_arch == 1:
        return archive_X[0], archive_X[0], archive_X[0]
    if n_arch == 2:
        return archive_X[0], archive_X[1], archive_X[0]

    cd = U.crowding_distance(archive_F)
    # Sort archive descending by crowding distance
    sorted_idx = np.argsort(-cd)

    alpha = archive_X[sorted_idx[0]]
    beta = archive_X[sorted_idx[1]]
    delta = archive_X[sorted_idx[2]]
    return alpha, beta, delta


def update_archive(archive_X, archive_F, archive_CV, new_X, new_F, new_CV, max_size):
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
        worst_idx = np.argmin(cd)
        arch_X = np.delete(arch_X, worst_idx, axis=0)
        arch_F = np.delete(arch_F, worst_idx, axis=0)
        arch_CV = np.delete(arch_CV, worst_idx, axis=0)

    return arch_X, arch_F, arch_CV


def initialize(problem, pack_size=50, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pack_size, rng)
    F, CV = problem.evaluate(X)
    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), np.empty(0),
        X, F, CV, archive_size
    )
    return X, F, CV, arch_X, arch_F, arch_CV


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, arch_X, arch_F, arch_CV, gen=0, max_gen=50, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N, d = X.shape

    # Select leaders
    alpha, beta, delta = select_leaders_from_archive(arch_X, arch_F, rng)
    if alpha is None:
        alpha = beta = delta = X[0]

    # a linearly decreases from 2 to 0
    a = 2.0 - 2.0 * (gen / max(1, max_gen))

    for i in range(N):
        r1 = rng.uniform(0.0, 1.0, size=d)
        r2 = rng.uniform(0.0, 1.0, size=d)
        A1 = 2.0 * a * r1 - a
        C1 = 2.0 * r2
        D_alpha = np.abs(C1 * alpha - X[i])
        X1 = alpha - A1 * D_alpha

        r1 = rng.uniform(0.0, 1.0, size=d)
        r2 = rng.uniform(0.0, 1.0, size=d)
        A2 = 2.0 * a * r1 - a
        C2 = 2.0 * r2
        D_beta = np.abs(C2 * beta - X[i])
        X2 = beta - A2 * D_beta

        r1 = rng.uniform(0.0, 1.0, size=d)
        r2 = rng.uniform(0.0, 1.0, size=d)
        A3 = 2.0 * a * r1 - a
        C3 = 2.0 * r2
        D_delta = np.abs(C3 * delta - X[i])
        X3 = delta - A3 * D_delta

        X_new = (X1 + X2 + X3) / 3.0
        X[i] = np.clip(X_new, problem.lower, problem.upper)
        if problem.var_type in ("int", "binary"):
            X[i] = np.round(X[i])

    F, CV = evaluate(problem, X)
    arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, X, F, CV, archive_size)
    return X, F, CV, arch_X, arch_F, arch_CV


def run(problem, pack_size=50, archive_size=100, n_gen=50, seed=42):
    rng = np.random.default_rng(seed)
    X, F, CV, arch_X, arch_F, arch_CV = initialize(
        problem, pack_size=pack_size, archive_size=archive_size, rng=rng
    )

    for gen in range(n_gen):
        X, F, CV, arch_X, arch_F, arch_CV = step(
            problem, X, F, CV, arch_X, arch_F, arch_CV,
            gen=gen, max_gen=n_gen, archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
