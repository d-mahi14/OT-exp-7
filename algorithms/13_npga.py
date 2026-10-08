"""
Algorithm 13: NPGA (Niched Pareto Genetic Algorithm)
Proposed by: Horn, Nafpliotis & Goldberg (1994)
Family: Non-Elitist MOEA (Part 2, slides 27-32)

Core idea:
Replace global fitness assignment entirely with Pareto tournaments:
Pick 2 candidates from the population and a comparison set of size t_dom.
If Pareto dominance directly separates the two candidates (or against the comparison set),
the dominant candidate wins. If neither dominates, break the tie using niche count
against the comparison set (the LESS crowded candidate wins):

    1. Pick 2 candidates i, j at random from population.
    2. Separately draw a comparison set C of size t_dom at random from population.
    3. If i dominates j -> select i; if j dominates i -> select j.
    4. If neither dominates:
       Compute niche count NC(i) and NC(j) against C:
       NC(candidate) = sum_{c in C} sh(d(candidate, c))
    5. Select candidate with LOWER niche count (less crowded).
    6. If still tied, pick at random.
    7. Crossover and mutation; non-elitist replacement.
"""

import numpy as np
import moo_utils as U
import operators as ops


def pareto_tournament(F, CV, i, j, comp_set, sigma_share=0.5, Fn=None):
    """
    Perform NPGA Pareto tournament between candidate i and candidate j
    using comparison set comp_set.
    """
    # 1. Direct dominance check between i and j (with CV handling)
    if CV is not None:
        i_beats_j = U.constrained_dominates(F[i], CV[i], F[j], CV[j])
        j_beats_i = U.constrained_dominates(F[j], CV[j], F[i], CV[i])
    else:
        i_beats_j = U.dominates(F[i], F[j])
        j_beats_i = U.dominates(F[j], F[i])

    if i_beats_j and not j_beats_i:
        return i
    if j_beats_i and not i_beats_j:
        return j

    # 2. Tie-break: compute niche count against comparison set in normalized space
    if Fn is None:
        Fn = ops.minmax_normalize(F)

    dist_i = np.linalg.norm(Fn[comp_set] - Fn[i], axis=1)
    dist_j = np.linalg.norm(Fn[comp_set] - Fn[j], axis=1)

    nc_i = float(ops.sharing_function(dist_i, sigma_share).sum())
    nc_j = float(ops.sharing_function(dist_j, sigma_share).sum())

    if nc_i < nc_j:
        return i
    elif nc_j < nc_i:
        return j
    else:
        return i if np.random.random() < 0.5 else j


def select_mating_pool(X, F, CV, t_dom=10, sigma_share=0.5, rng=None):
    """
    Select N parents using NPGA Pareto tournament selection.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)
    Fn = ops.minmax_normalize(F)
    selected_indices = np.empty(N, dtype=int)

    for idx in range(N):
        i, j = rng.integers(N, size=2)
        # Draw comparison set of size t_dom
        comp_size = min(t_dom, N)
        comp_set = rng.choice(N, size=comp_size, replace=False)
        selected_indices[idx] = pareto_tournament(F, CV, i, j, comp_set, sigma_share, Fn)

    return selected_indices


def initialize(problem, pop_size=100, rng=None):
    """
    Initialize random population.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return problem.random_solutions(pop_size, rng)


def evaluate(problem, X):
    """
    Evaluate objective vectors and constraint violations.
    """
    return problem.evaluate(X)


def step(problem, X, F, CV, t_dom=10, sigma_share=0.5, pc=0.9, pm=None, rng=None):
    """
    Perform one NPGA generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # 1. Tournament selection
    selected_idx = select_mating_pool(X, F, CV, t_dom=t_dom, sigma_share=sigma_share, rng=rng)
    parents = X[selected_idx]

    # 2. Variation
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

    X_next = np.array(offspring[:N])
    if problem.var_type == "int":
        X_next = np.round(X_next)
    F_next, CV_next = evaluate(problem, X_next)
    return X_next, F_next, CV_next


def run(problem, pop_size=100, n_gen=50, t_dom=10, sigma_share=0.5, pc=0.9, pm=None, seed=42):
    """
    Run NPGA optimization.
    """
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    archive_X, archive_F = X.copy(), F.copy()

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, t_dom=t_dom, sigma_share=sigma_share, pc=pc, pm=pm, rng=rng)
        archive_X = np.vstack([archive_X, X])
        archive_F = np.vstack([archive_F, F])
        nd_idx = U.pareto_front(archive_F)
        archive_X = archive_X[nd_idx]
        archive_F = archive_F[nd_idx]

    nd_final = U.pareto_front(archive_F)
    return archive_X[nd_final], archive_F[nd_final]
