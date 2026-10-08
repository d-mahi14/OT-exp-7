"""
Algorithm 35: R-NSGA-II (Reference Point-Based NSGA-II)
Proposed by: Deb, Sundar, Bhaskara & Chaudhuri (2006)
Family: Preference-Based MOEA (Part 5E, slides 20-22)

Core idea:
Focuses search around a decision-maker's preferred trade-off point using ε-clearing:
1. DM specifies a reference point z_ref in objective space.
2. Min-max normalize objectives across the population.
3. Compute Euclidean distance d_i from each solution to the normalized reference point.
4. Non-dominated sort into fronts F_1, F_2, ...
5. Within each front (or across the population), sort ascending by distance d_i.
6. ε-clearing rule:
   Accept solution only if it is >= ε_clearing away (in normalized space) from
   all already-accepted solutions, preserving a spread around the preference.
7. If clearing leaves fewer than N solutions, backfill by pure distance order.
"""

import numpy as np
import moo_utils as U
import operators as ops


def compute_normalized_reference_distances(F, ref_point, f_min=None, f_max=None):
    """
    Min-max normalize objectives and compute distance to normalized reference point.
    """
    if f_min is None:
        f_min = np.min(F, axis=0)
    if f_max is None:
        f_max = np.max(F, axis=0)

    span = f_max - f_min
    span = np.where(span <= 0, 1.0, span)

    Fn = (F - f_min) / span
    z_norm = (ref_point - f_min) / span

    dists = np.linalg.norm(Fn - z_norm, axis=1)
    return dists, Fn


def epsilon_clearing_selection(X_comb, F_comb, CV_comb, ref_point, target_size, eps_clearing=0.15):
    """
    R-NSGA-II ε-clearing survivor selection.
    """
    N_comb = len(X_comb)
    if N_comb <= target_size:
        return X_comb, F_comb, CV_comb

    # 1. Non-dominated sort
    fronts = U.non_dominated_sort(F_comb, CV_comb)

    # 2. Distances to reference point
    dists, Fn = compute_normalized_reference_distances(F_comb, ref_point)

    accepted = []

    # Front-by-front selection with clearing
    for front in fronts:
        # Sort front by ascending distance to reference
        sorted_front = sorted(front, key=lambda idx: dists[idx])

        for cand in sorted_front:
            if len(accepted) >= target_size:
                break
            # Check clearing against all currently accepted
            if len(accepted) == 0:
                accepted.append(cand)
            else:
                acc_Fn = Fn[accepted]
                cand_Fn = Fn[cand]
                dist_to_acc = np.linalg.norm(acc_Fn - cand_Fn, axis=1)
                if np.all(dist_to_acc >= eps_clearing):
                    accepted.append(cand)

        if len(accepted) >= target_size:
            break

    # If clearing left fewer than target_size, backfill by pure distance
    if len(accepted) < target_size:
        all_sorted = np.argsort(dists)
        for idx in all_sorted:
            if idx not in accepted:
                accepted.append(idx)
                if len(accepted) == target_size:
                    break

    accepted = np.array(accepted[:target_size], dtype=int)
    return X_comb[accepted], F_comb[accepted], CV_comb[accepted]


def initialize(problem, pop_size=50, ref_point=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pop_size, rng)
    if ref_point is None:
        if hasattr(problem, "preference_point"):
            # Natural to min form
            ref_point = problem.preference_point.copy()
            ref_point[problem.maximize] *= -1.0
        else:
            ref_point = problem.ideal.copy()
    return X, ref_point


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, ref_point, eps_clearing=0.15, pc=0.9, pm=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # Tournament selection favoring lower distance to reference
    dists, _ = compute_normalized_reference_distances(F, ref_point)
    mating_idx = np.empty(N, dtype=int)
    for i in range(N):
        a, b = rng.integers(N, size=2)
        mating_idx[i] = a if dists[a] <= dists[b] else b
    parents = X[mating_idx]

    offspring = []
    for i in range(0, N, 2):
        p1 = parents[i]
        p2 = parents[(i + 1) % N]
        c1, c2 = ops.sbx_crossover(p1, p2, problem.lower, problem.upper, rng, pc=pc)
        c1 = ops.polynomial_mutation(c1, problem.lower, problem.upper, rng, pm=pm)
        c2 = ops.polynomial_mutation(c2, problem.lower, problem.upper, rng, pm=pm)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    X_off = np.array(offspring[:N])
    if problem.var_type == "int":
        X_off = np.round(X_off)
    F_off, CV_off = evaluate(problem, X_off)

    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    CV_comb = np.concatenate([CV, CV_off])

    return epsilon_clearing_selection(X_comb, F_comb, CV_comb, ref_point, target_size=N, eps_clearing=eps_clearing)


def run(problem, pop_size=50, n_gen=50, ref_point=None, eps_clearing=0.15, pc=0.9, pm=None, seed=42):
    rng = np.random.default_rng(seed)
    X, ref_point = initialize(problem, pop_size=pop_size, ref_point=ref_point, rng=rng)
    F, CV = evaluate(problem, X)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, ref_point, eps_clearing=eps_clearing, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
