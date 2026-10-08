"""
Algorithm 20: MOEA/D-ACO (MOEA/D with Ant Colony Optimization)
Proposed by: Ke, Zhang & Battiti (2013)
Family: Elitist, Decomposition-Based MOEA (Part 3, slides 51-57)

Core idea:
Replaces MOEA/D's genetic variation with Ant Colony Optimization (ACO) route
construction, maintaining pheromone trails per subproblem (or per objective) --
specifically designed for discrete / combinatorial MOO problems such as P8 (Bi-Objective TSP):

    1. Decompose problem with N weight vectors lambda_1, ..., lambda_N.
    2. Maintain pheromone matrices tau_ij per subproblem (or joint pheromone).
    3. Each generation, ants construct discrete solutions (e.g. TSP permutations) using:
       p_ij proportional to (tau_ij)^alpha * (eta_ij)^beta
       where eta_ij is heuristic desirability (e.g. 1 / distance).
    4. Evaluate objectives (distance, time).
    5. Update subproblem best solutions and neighborhoods using scalar decomposition.
    6. Update pheromones:
       tau_ij <- (1 - rho) * tau_ij + sum Delta tau_ij
       where only non-dominated or neighborhood-best solutions deposit pheromone.
"""

import numpy as np
import moo_utils as U
import operators as ops


def construct_ant_tour(n_cities, tau, dist_matrix, alpha=1.0, beta=2.0, rng=None):
    """
    Construct a TSP tour using ACO probabilistic transition.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    unvisited = list(range(n_cities))
    start_city = rng.integers(n_cities)
    tour = [start_city]
    unvisited.remove(start_city)

    current = start_city
    while unvisited:
        # Compute probabilities for remaining unvisited cities
        probs = np.zeros(len(unvisited))
        for idx, city in enumerate(unvisited):
            d = dist_matrix[current, city]
            eta = 1.0 / max(d, 1e-6)
            tau_val = max(tau[current, city], 1e-6)
            probs[idx] = (tau_val ** alpha) * (eta ** beta)

        sum_p = np.sum(probs)
        if sum_p <= 0 or np.isnan(sum_p):
            next_idx = rng.integers(len(unvisited))
        else:
            probs /= sum_p
            next_idx = rng.choice(len(unvisited), p=probs)

        current = unvisited[next_idx]
        tour.append(current)
        del unvisited[next_idx]

    return np.array(tour, dtype=int)


def initialize(problem, n_subproblems=20, T=5, rng=None):
    """
    Initialize weight vectors, neighborhoods, pheromone matrices, and initial tours.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    weights = ops.weight_grid(problem.n_obj, n_weights=n_subproblems)
    N = len(weights)
    T = min(T, N)
    dists = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighborhoods = np.argsort(dists, axis=1)[:, :T]

    n_cities = problem.n_var
    # Shared pheromone matrix or one per subproblem
    tau = np.ones((n_cities, n_cities), dtype=float)

    # Initial random tours
    X = np.array([rng.permutation(n_cities) for _ in range(N)])
    F, CV = problem.evaluate(X)
    z_ideal = np.min(F, axis=0)

    return weights, neighborhoods, tau, X, F, CV, z_ideal


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, weights, neighborhoods, tau, X, F, CV, z_ideal,
         alpha=1.0, beta=2.0, rho=0.3, Q=10.0, rng=None):
    """
    Perform one iteration of MOEA/D-ACO.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)
    n_cities = problem.n_var

    # Use distance matrix if available (P8 has dist)
    if hasattr(problem, "dist"):
        dist_mat = problem.dist
    elif hasattr(problem, "D"):
        dist_mat = problem.D
    else:
        dist_mat = np.ones((n_cities, n_cities))

    # 1. Ant tour construction for each subproblem
    new_tours = []
    for i in range(N):
        tour = construct_ant_tour(n_cities, tau, dist_mat, alpha=alpha, beta=beta, rng=rng)
        new_tours.append(tour)

    new_X = np.array(new_tours)
    new_F, new_CV = problem.evaluate(new_X)

    # 2. Update ideal point
    z_ideal = np.minimum(z_ideal, np.min(new_F, axis=0))

    # 3. Update neighborhood solutions using weighted sum / Tchebycheff
    for i in range(N):
        fy = new_F[i]
        cvy = new_CV[i]
        y = new_X[i]
        for j in neighborhoods[i]:
            diff_y = np.abs(fy - z_ideal)
            diff_x = np.abs(F[j] - z_ideal)
            cost_y = np.max(weights[j] * diff_y) + 1e6 * cvy
            cost_x = np.max(weights[j] * diff_x) + 1e6 * CV[j]
            if cost_y <= cost_x:
                X[j] = y.copy()
                F[j] = fy.copy()
                CV[j] = cvy

    # 4. Pheromone evaporation
    tau *= (1.0 - rho)

    # 5. Pheromone deposit by non-dominated tours
    nd_idx = U.pareto_front(F, CV)
    for idx in nd_idx:
        tour = X[idx]
        cost = np.sum(F[idx])  # scalar scale for deposit
        deposit = Q / max(cost, 1e-6)
        for c in range(n_cities):
            u = tour[c]
            v = tour[(c + 1) % n_cities]
            tau[u, v] += deposit
            tau[v, u] += deposit

    # Clamp pheromone values for stability
    tau = np.clip(tau, 0.01, 100.0)

    return tau, X, F, CV, z_ideal


def run(problem, n_subproblems=20, T=5, n_gen=30, alpha=1.0, beta=2.0, rho=0.3, Q=10.0, seed=42):
    """
    Run MOEA/D-ACO optimization.
    """
    rng = np.random.default_rng(seed)
    weights, neighborhoods, tau, X, F, CV, z_ideal = initialize(
        problem, n_subproblems=n_subproblems, T=T, rng=rng
    )

    for gen in range(n_gen):
        tau, X, F, CV, z_ideal = step(
            problem, weights, neighborhoods, tau, X, F, CV, z_ideal,
            alpha=alpha, beta=beta, rho=rho, Q=Q, rng=rng
        )

    nd_final = U.pareto_front(F, CV)
    return X[nd_final], F[nd_final]
