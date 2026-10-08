"""
Algorithm 12: NSGA (Non-Dominated Sorting Genetic Algorithm - Original)
Proposed by: Srinivas & Deb (1994)
Family: Non-Elitist MOEA (Part 2, slides 21-26)

Core idea:
Peel the population into hierarchical fronts by repeated non-dominated sorting
(Front 1 = non-dominated, Front 2 = non-dominated when Front 1 is removed, etc.).
Assign a dummy fitness to each front, then apply fitness sharing strictly WITHIN
each front so individuals spread out across each front without blending ranks:

    1. Sort population into fronts F_1, F_2, ... using non-dominated sorting.
    2. Assign initial dummy fitness F_max to Front 1.
    3. Within Front 1, compute pairwise distances in normalized objective space:
       sh(d_ij) = 1 - (d_ij / sigma_share)^2 if d_ij < sigma_share else 0
       NC(i) = sum_{j in Front 1} sh(d_ij)
       Shared fitness: F_shared(i) = F_dummy / NC(i)
    4. For subsequent Front k:
       F_dummy = min(F_shared of Front k-1) - delta_F (ensures strict tiering)
       Compute sharing strictly within Front k.
    5. Selection via roulette wheel on F_shared.
    6. Crossover and mutation; non-elitist replacement.
"""

import numpy as np
import moo_utils as U
import operators as ops


def assign_nsga_shared_fitness(F, CV=None, sigma_share=0.5, f_max=100.0, delta_f=5.0):
    """
    Assign dummy fitness and within-front shared fitness for NSGA.
    """
    N = len(F)
    fronts = U.non_dominated_sort(F, CV)
    shared_fitness = np.zeros(N, dtype=float)

    Fn = ops.minmax_normalize(F)
    current_dummy = float(f_max)

    for front in fronts:
        if len(front) == 0:
            continue
        front_Fn = Fn[front]
        # Compute niche counts within this front only
        if len(front) == 1:
            niche_counts = np.array([1.0])
        else:
            D = np.linalg.norm(front_Fn[:, None, :] - front_Fn[None, :, :], axis=2)
            sh = ops.sharing_function(D, sigma_share)
            niche_counts = sh.sum(axis=1)

        f_shared_front = current_dummy / niche_counts
        shared_fitness[front] = f_shared_front

        # Set next dummy fitness below the minimum of this front's shared fitness
        min_shared = np.min(f_shared_front)
        current_dummy = max(1.0, min_shared - delta_f)

    return shared_fitness, fronts


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


def step(problem, X, F, CV, sigma_share=0.5, pc=0.9, pm=None, rng=None):
    """
    Perform one NSGA generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # 1. Fronts and shared fitness
    shared_fit, _ = assign_nsga_shared_fitness(F, CV, sigma_share=sigma_share)

    # 2. Selection via roulette wheel
    selected_idx = ops.roulette_wheel(shared_fit, N, rng)
    parents = X[selected_idx]

    # 3. Variation
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


def run(problem, pop_size=100, n_gen=50, sigma_share=0.5, pc=0.9, pm=None, seed=42):
    """
    Run NSGA optimization.
    """
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, sigma_share=sigma_share, pc=pc, pm=pm, rng=rng)

    nd_final = U.pareto_front(F, CV)
    return X[nd_final], F[nd_final]
