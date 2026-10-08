"""
Algorithm 18: MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)
Proposed by: Zhang & Li (2007)
Family: Elitist, Decomposition-Based MOEA (Part 3, slides 36-43)

Core idea:
Decomposes the multiobjective optimization problem into N scalar single-objective
subproblems using weight vectors lambda_1, ..., lambda_N, and optimizes them
collaboratively by sharing information among neighboring subproblems:

    1. Initialize N weight vectors lambda_1, ..., lambda_N spread across the unit simplex.
    2. Compute neighborhood B(i) of the T closest weight vectors for each subproblem i.
    3. Initialize population x_1, ..., x_N (one per subproblem) and ideal point z*.
    4. For each subproblem i:
       a. Select mating parents from neighborhood B(i).
       b. Generate offspring y via crossover (SBX) and mutation (polynomial).
       c. Update ideal point z*_m = min(z*_m, f_m(y)).
       d. For each neighbor j in B(i):
          If Tchebycheff cost g(y | lambda_j, z*) <= g(x_j | lambda_j, z*),
          replace x_j with y.
    5. Maintain external archive of non-dominated solutions.
"""

import numpy as np
import moo_utils as U
import operators as ops


def tchebycheff_subproblem(objectives, weights, z_ideal):
    """
    Tchebycheff scalar decomposition function:
    g(x | lambda, z*) = max_m [ lambda_m * |f_m(x) - z*_m| ]
    """
    diff = np.abs(np.asarray(objectives, float) - np.asarray(z_ideal, float))
    w = np.asarray(weights, float)
    # Avoid zero weights completely nullifying objectives
    w = np.where(w < 1e-6, 1e-6, w)
    return float(np.max(w * diff))


def compute_neighborhoods(weights, T=20):
    """
    Find indices of T closest weight vectors for each subproblem.
    """
    N = len(weights)
    T = min(T, N)
    dists = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    B = np.argsort(dists, axis=1)[:, :T]
    return B


def initialize(problem, n_subproblems=100, T=20, rng=None):
    """
    Initialize weight vectors, neighborhoods, initial population, and ideal point.
    """
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


def step(problem, weights, neighborhoods, X, F, CV, z_ideal, pc=0.9, pm=None, rng=None):
    """
    Perform one generation of MOEA/D updates across all subproblems.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    for i in range(N):
        # Select parents from neighborhood B(i)
        nb = neighborhoods[i]
        p_idx = rng.choice(nb, size=2, replace=False)
        p1, p2 = X[p_idx[0]], X[p_idx[1]]

        # Crossover & mutation
        c1, _ = ops.sbx_crossover(p1, p2, problem.lower, problem.upper, rng, pc=pc)
        y = ops.polynomial_mutation(c1, problem.lower, problem.upper, rng, pm=pm)
        if problem.var_type == "int":
            y = np.round(y)

        fy, cvy = problem.evaluate(y)
        fy = fy[0] if fy.ndim > 1 else fy
        cvy = cvy[0] if np.ndim(cvy) > 0 else cvy

        # Update ideal point
        z_ideal = np.minimum(z_ideal, fy)

        # Update neighbors in B(i)
        cost_y = tchebycheff_subproblem(fy, weights[i], z_ideal) + 1e6 * cvy
        for j in nb:
            cost_yj = tchebycheff_subproblem(fy, weights[j], z_ideal) + 1e6 * cvy
            cost_xj = tchebycheff_subproblem(F[j], weights[j], z_ideal) + 1e6 * CV[j]
            if cost_yj <= cost_xj:
                X[j] = y.copy()
                F[j] = fy.copy()
                CV[j] = cvy

    return X, F, CV, z_ideal


def run(problem, n_subproblems=100, T=20, n_gen=50, pc=0.9, pm=None, seed=42):
    """
    Run MOEA/D optimization.
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
            problem, weights, neighborhoods, X, F, CV, z_ideal, pc=pc, pm=pm, rng=rng
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
