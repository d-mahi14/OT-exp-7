"""
Algorithm 19: MOEA/D-DE (MOEA/D with Differential Evolution)
Proposed by: Li & Zhang (2009)
Family: Elitist, Decomposition-Based MOEA (Part 3, slides 44-50)

Core idea:
Enhances MOEA/D by replacing standard genetic crossover with Differential Evolution (DE)
operators (DE mutation + binomial crossover), which significantly accelerates convergence
on continuous, real-valued multiobjective landscapes:

    1. Subproblems and neighborhoods B(i) structured identical to MOEA/D.
    2. For subproblem i:
       a. Pick 3 distinct neighbors r1, r2, r3 from B(i).
       b. DE mutation: V = X_r1 + F_de * (X_r2 - X_r3)
       c. Binomial crossover with target X_i:
          U_j = V_j if rand_j < CR (or forced dimension) else X_{i, j}
       d. Polynomial mutation on U; clip to bounds.
       e. Update ideal point z*.
       f. Update neighbors in B(i) using Tchebycheff aggregation.
"""

import numpy as np
import moo_utils as U
import operators as ops


def tchebycheff_subproblem(objectives, weights, z_ideal):
    """
    Tchebycheff scalar decomposition function.
    """
    diff = np.abs(np.asarray(objectives, float) - np.asarray(z_ideal, float))
    w = np.asarray(weights, float)
    w = np.where(w < 1e-6, 1e-6, w)
    return float(np.max(w * diff))


def compute_neighborhoods(weights, T=20):
    N = len(weights)
    T = min(T, N)
    dists = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    return np.argsort(dists, axis=1)[:, :T]


def initialize(problem, n_subproblems=100, T=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    weights = ops.weight_grid(problem.n_obj, n_weights=n_subproblems)
    N = len(weights)
    T = min(T, N)
    neighborhoods = compute_neighborhoods(weights, T=T)

    X = problem.random_solutions(N, rng)
    F, CV = problem.evaluate(X)
    z_ideal = np.min(F, axis=0)

    return weights, neighborhoods, X, F, CV, z_ideal


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, weights, neighborhoods, X, F, CV, z_ideal, F_de=0.5, CR=0.9, pm=None, rng=None):
    """
    Perform one generation of MOEA/D-DE.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    for i in range(N):
        nb = neighborhoods[i]
        # Choose 3 distinct vectors from neighborhood
        if len(nb) >= 3:
            r1, r2, r3 = rng.choice(nb, size=3, replace=False)
        else:
            r1, r2, r3 = rng.choice(N, size=3, replace=False)

        # 1. DE Mutation
        v = ops.de_mutation(X[r1], X[r2], X[r3], F=F_de)

        # 2. DE Binomial Crossover with target X[i]
        u = ops.de_binomial_crossover(X[i], v, CR=CR, rng=rng)

        # 3. Polynomial mutation
        y = ops.polynomial_mutation(u, problem.lower, problem.upper, rng, pm=pm)
        if problem.var_type == "int":
            y = np.round(y)

        # 4. Evaluate
        fy, cvy = problem.evaluate(y)
        fy = fy[0] if fy.ndim > 1 else fy
        cvy = cvy[0] if np.ndim(cvy) > 0 else cvy

        # 5. Update ideal point
        z_ideal = np.minimum(z_ideal, fy)

        # 6. Update neighborhood
        cost_y = tchebycheff_subproblem(fy, weights[i], z_ideal) + 1e6 * cvy
        for j in nb:
            cost_yj = tchebycheff_subproblem(fy, weights[j], z_ideal) + 1e6 * cvy
            cost_xj = tchebycheff_subproblem(F[j], weights[j], z_ideal) + 1e6 * CV[j]
            if cost_yj <= cost_xj:
                X[j] = y.copy()
                F[j] = fy.copy()
                CV[j] = cvy

    return X, F, CV, z_ideal


def run(problem, n_subproblems=100, T=20, n_gen=50, F_de=0.5, CR=0.9, pm=None, seed=42):
    """
    Run MOEA/D-DE optimization.
    """
    rng = np.random.default_rng(seed)
    weights, neighborhoods, X, F, CV, z_ideal = initialize(
        problem, n_subproblems=n_subproblems, T=T, rng=rng
    )

    archive_X = X.copy()
    archive_F = F.copy()
    archive_CV = CV.copy()

    for gen in range(n_gen):
        X, F, CV, z_ideal = step(
            problem, weights, neighborhoods, X, F, CV, z_ideal,
            F_de=F_de, CR=CR, pm=pm, rng=rng
        )
        archive_X = np.vstack([archive_X, X])
        archive_F = np.vstack([archive_F, F])
        archive_CV = np.concatenate([archive_CV, CV])
        nd_idx = U.pareto_front(archive_F, archive_CV)
        archive_X = archive_X[nd_idx]
        archive_F = archive_F[nd_idx]
        archive_CV = archive_CV[nd_idx]

    nd_final = U.pareto_front(archive_F, archive_CV)
    return archive_X[nd_final], archive_F[nd_final]
