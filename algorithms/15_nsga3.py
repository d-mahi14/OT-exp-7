"""
Algorithm 15: NSGA-III (Non-Dominated Sorting Genetic Algorithm III)
Proposed by: Deb & Jain (2014)
Family: Elitist, Many-Objective MOEA (Part 3, slides 12-19; Part 5E, slides 5-7)

Core idea:
Replaces crowding distance with structured reference points on a normalized
hyperplane to maintain selection pressure and diversity in many-objective problems (k >= 3):

    1. Generate H = C(k+p-1, p) structured reference points on the unit simplex (Das-Dennis).
    2. Population size N ≈ H.
    3. Merge parents + offspring (size 2N); sort into fronts F_1, F_2, ...
    4. Compute ideal point z* = min F across all individuals.
    5. Translate and normalize objectives: f'_m = (f_m - z*_m) / (nadir_m - z*_m).
    6. Associate each individual with its nearest reference direction by perpendicular distance:
       d_perp(s, w) = || s - (s . w) w ||   (where w = r / ||r||).
    7. Fill next generation front-by-front. In the last partially accepted front,
       use niche preservation: prioritize reference points with minimum niche count rho_j.
"""

import numpy as np
import moo_utils as U
import operators as ops


def generate_reference_points(k, p=4):
    """
    Generate Das-Dennis reference points on unit simplex.
    """
    return U.das_dennis(k, p)


def associate_to_reference_points(F_norm, ref_points):
    """
    Associate each normalized objective vector with the nearest reference direction.
    Returns:
        associations : (N,) index of nearest reference point
        perpendicular_distances : (N,) distance to that reference direction
    """
    N = len(F_norm)
    H = len(ref_points)

    # Unit direction vectors
    norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
    W = ref_points / np.where(norms == 0, 1.0, norms)  # (H, k)

    associations = np.zeros(N, dtype=int)
    d_perp = np.zeros(N, dtype=float)

    for i in range(N):
        s = F_norm[i]
        # Projections onto all reference lines: (H,)
        projs = np.dot(W, s)
        # Perpendicular distances: ||s - proj * w||
        diffs = s[None, :] - projs[:, None] * W
        dists = np.linalg.norm(diffs, axis=1)

        best_ref = np.argmin(dists)
        associations[i] = best_ref
        d_perp[i] = dists[best_ref]

    return associations, d_perp


def nsga3_survivor_selection(X_comb, F_comb, CV_comb, ref_points, pop_size):
    """
    NSGA-III niche-preservation environmental selection.
    """
    fronts = U.non_dominated_sort(F_comb, CV_comb)
    survivors = []
    last_front_idx = -1

    for idx, front in enumerate(fronts):
        if len(survivors) + len(front) <= pop_size:
            survivors.extend(front)
        else:
            last_front_idx = idx
            break

    if len(survivors) == pop_size or last_front_idx == -1:
        chosen = np.array(survivors, dtype=int)
        return X_comb[chosen], F_comb[chosen], CV_comb[chosen]

    # Normalize objectives using ideal and estimated nadir
    z_ideal = np.min(F_comb, axis=0)
    z_max = np.max(F_comb, axis=0)
    denom = z_max - z_ideal
    denom = np.where(denom <= 0, 1.0, denom)
    F_norm = (F_comb - z_ideal) / denom

    # Associate all individuals with reference points
    associations, d_perp = associate_to_reference_points(F_norm, ref_points)

    # Tally niche counts for already accepted individuals
    H = len(ref_points)
    niche_counts = np.zeros(H, dtype=int)
    for idx in survivors:
        niche_counts[associations[idx]] += 1

    last_front = fronts[last_front_idx]
    last_front_set = set(last_front)
    needed = pop_size - len(survivors)

    while needed > 0 and len(last_front_set) > 0:
        # Find reference point with minimum niche count that has members in last_front
        candidate_refs = []
        for r_idx in range(H):
            members = [i for i in last_front_set if associations[i] == r_idx]
            if len(members) > 0:
                candidate_refs.append(r_idx)

        if not candidate_refs:
            # Fallback: take remaining from last front
            chosen_fallback = list(last_front_set)[:needed]
            survivors.extend(chosen_fallback)
            break

        min_rho = min(niche_counts[r] for r in candidate_refs)
        min_refs = [r for r in candidate_refs if niche_counts[r] == min_rho]
        chosen_ref = min_refs[0]

        members = [i for i in last_front_set if associations[i] == chosen_ref]
        if niche_counts[chosen_ref] == 0:
            # Choose individual with minimum perpendicular distance
            best_ind = min(members, key=lambda i: d_perp[i])
        else:
            # Choose at random
            best_ind = members[0]

        survivors.append(best_ind)
        last_front_set.remove(best_ind)
        niche_counts[chosen_ref] += 1
        needed -= 1

    chosen = np.array(survivors[:pop_size], dtype=int)
    return X_comb[chosen], F_comb[chosen], CV_comb[chosen]


def initialize(problem, p=4, rng=None):
    """
    Initialize population and reference points.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    ref_points = generate_reference_points(problem.n_obj, p=p)
    N = len(ref_points)
    # Ensure N is multiple of 4 or at least 12
    N = max(N, 12)
    X = problem.random_solutions(N, rng)
    return ref_points, X


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, ref_points, pc=0.9, pm=None, rng=None):
    """
    Perform one NSGA-III generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # Tournament selection
    fronts = U.non_dominated_sort(F, CV)
    ranks = np.zeros(N, dtype=int)
    for r, fr in enumerate(fronts):
        ranks[fr] = r

    # Binary tournament on rank
    mating_idx = np.empty(N, dtype=int)
    for i in range(N):
        a, b = rng.integers(N, size=2)
        mating_idx[i] = a if ranks[a] <= ranks[b] else b
    parents = X[mating_idx]

    # Generate offspring
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
    F_off, CV_off = evaluate(problem, X_off)

    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    CV_comb = np.concatenate([CV, CV_off])

    return nsga3_survivor_selection(X_comb, F_comb, CV_comb, ref_points, N)


def run(problem, p=4, n_gen=50, pc=0.9, pm=None, seed=42):
    """
    Run NSGA-III algorithm.
    """
    rng = np.random.default_rng(seed)
    ref_points, X = initialize(problem, p=p, rng=rng)
    F, CV = evaluate(problem, X)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, ref_points, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
