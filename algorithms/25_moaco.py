"""
Algorithm 25: MOACO (Multi-Objective Ant Colony Optimization)
Family: Swarm (Part 4, slides 35-41)

Core idea:
Designed for discrete / combinatorial MOO (such as P8 Bi-objective TSP):
Maintains two separate pheromone matrices (tau_1 for distance, tau_2 for time):

    1. Transition probability for ant moving from city i to city j:
       p_ij proportional to (tau_1_ij)^alpha * (tau_2_ij)^alpha * (eta_ij)^beta
       where eta_ij = 1 / distance_ij.
    2. Ants construct complete permutations / tours.
    3. Evaluate objectives f_1 (distance) and f_2 (time).
    4. Non-dominated tours enter an external Pareto archive.
    5. Pheromone evaporation and deposit:
       tau_k_ij <- (1 - rho) * tau_k_ij + sum_{non-dom} Q / f_k(route)
       Dominated routes deposit zero pheromone, steering future ants toward trade-offs.
"""

import numpy as np
import moo_utils as U


def construct_ant_route(n_cities, tau1, tau2, dist_matrix, alpha=1.0, beta=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    unvisited = list(range(n_cities))
    start_city = rng.integers(n_cities)
    route = [start_city]
    unvisited.remove(start_city)

    current = start_city
    while unvisited:
        probs = np.zeros(len(unvisited))
        for idx, city in enumerate(unvisited):
            d = dist_matrix[current, city]
            eta = 1.0 / max(d, 1e-6)
            t1 = max(tau1[current, city], 1e-6)
            t2 = max(tau2[current, city], 1e-6)
            probs[idx] = (t1 ** alpha) * (t2 ** alpha) * (eta ** beta)

        sum_p = np.sum(probs)
        if sum_p <= 0 or np.isnan(sum_p):
            next_idx = rng.integers(len(unvisited))
        else:
            probs /= sum_p
            next_idx = rng.choice(len(unvisited), p=probs)

        current = unvisited[next_idx]
        route.append(current)
        del unvisited[next_idx]

    return np.array(route, dtype=int)


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


def initialize(problem, n_ants=20, archive_size=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_cities = problem.n_var
    tau1 = np.ones((n_cities, n_cities), dtype=float)
    tau2 = np.ones((n_cities, n_cities), dtype=float)

    X = np.array([rng.permutation(n_cities) for _ in range(n_ants)])
    F, CV = problem.evaluate(X)

    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, n_cities), dtype=int), np.empty((0, problem.n_obj)), np.empty(0),
        X, F, CV, archive_size
    )
    return tau1, tau2, arch_X, arch_F, arch_CV


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, tau1, tau2, arch_X, arch_F, arch_CV, n_ants=20,
         alpha=1.0, beta=1.0, rho=0.3, Q=10.0, archive_size=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_cities = problem.n_var

    dist_mat = problem.dist if hasattr(problem, "dist") else np.ones((n_cities, n_cities))

    # 1. Ants construct routes
    routes = [construct_ant_route(n_cities, tau1, tau2, dist_mat, alpha=alpha, beta=beta, rng=rng)
              for _ in range(n_ants)]
    routes = np.array(routes)
    F, CV = evaluate(problem, routes)

    # 2. Update archive
    arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, routes, F, CV, archive_size)

    # 3. Pheromone evaporation
    tau1 *= (1.0 - rho)
    tau2 *= (1.0 - rho)

    # 4. Deposit on non-dominated routes in archive
    for idx in range(len(arch_X)):
        tour = arch_X[idx]
        f1_val = max(float(arch_F[idx, 0]), 1e-3)
        f2_val = max(float(arch_F[idx, 1]), 1e-3)
        dep1 = Q / f1_val
        dep2 = Q / f2_val

        for c in range(n_cities):
            u = tour[c]
            v = tour[(c + 1) % n_cities]
            tau1[u, v] += dep1
            tau1[v, u] += dep1
            tau2[u, v] += dep2
            tau2[v, u] += dep2

    tau1 = np.clip(tau1, 0.01, 100.0)
    tau2 = np.clip(tau2, 0.01, 100.0)

    return tau1, tau2, arch_X, arch_F, arch_CV


def run(problem, n_ants=20, archive_size=50, n_gen=30, alpha=1.0, beta=1.0, rho=0.3, Q=10.0, seed=42):
    rng = np.random.default_rng(seed)
    tau1, tau2, arch_X, arch_F, arch_CV = initialize(
        problem, n_ants=n_ants, archive_size=archive_size, rng=rng
    )

    for gen in range(n_gen):
        tau1, tau2, arch_X, arch_F, arch_CV = step(
            problem, tau1, tau2, arch_X, arch_F, arch_CV,
            n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, Q=Q,
            archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
