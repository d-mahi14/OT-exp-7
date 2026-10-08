"""
Algorithm 32: IBEA (Indicator-Based Evolutionary Algorithm)
Proposed by: Zitzler & Künzli (2004)
Family: Indicator-Based MOEA (Part 5E, slides 11-13)

Core idea:
Fitness is defined directly by a binary quality indicator (the additive ε-indicator)
measured pairwise against all other solutions in the population:

    1. Pairwise additive ε-indicator in normalized objective space:
           I(a, b) = max_{m=1..k} (a_m - b_m)
       I(a, b) <= 0 indicates that a weakly dominates b.
    2. Fitness of individual i:
           F(i) = sum_{j != i} -exp(-I(j, i) / kappa)
       (Lower/more negative is better; higher/less negative is worse).
    3. Environmental selection:
       Repeatedly remove the individual with the largest (worst) F(i).
       After deleting the worst individual w, update remaining fitnesses:
           F(j) <- F(j) + exp(-I(w, j) / kappa)
       until target population size N is reached.
    4. Variation generates offspring for the next generation.
"""

import numpy as np
import moo_utils as U
import operators as ops


def compute_epsilon_indicator_matrix(F_norm):
    """
    Compute pairwise additive epsilon-indicator matrix:
    I[i, j] = max_m (F_norm[i, m] - F_norm[j, m])
    """
    N = len(F_norm)
    # Shape (N, N, k)
    diff = F_norm[:, None, :] - F_norm[None, :, :]
    I_mat = np.max(diff, axis=2)
    return I_mat


def compute_ibea_fitness(I_mat, kappa=0.05):
    """
    F(i) = sum_{j != i} -exp(-I(j, i) / kappa)
    (Higher is worse / first candidate to drop).
    """
    N = len(I_mat)
    # Clip exponent to avoid numerical overflow/underflow
    exponent = -I_mat / max(kappa, 1e-6)
    exponent = np.clip(exponent, -50.0, 50.0)
    exp_mat = np.exp(exponent)
    np.fill_diagonal(exp_mat, 0.0)
    # F[i] = sum_{j != i} -exp(-I(j, i) / kappa)
    # Notice I(j, i) is row j, column i of I_mat
    fitness = -np.sum(exp_mat, axis=0)
    return fitness


def environmental_selection(X_comb, F_comb, CV_comb, target_size, kappa=0.05):
    """
    Iteratively remove worst individuals until target_size is reached.
    """
    N_comb = len(X_comb)
    if N_comb <= target_size:
        return X_comb, F_comb, CV_comb

    # Normalize objectives
    z_min = np.min(F_comb, axis=0)
    z_max = np.max(F_comb, axis=0)
    span = z_max - z_min
    span = np.where(span <= 0, 1.0, span)
    F_norm = (F_comb - z_min) / span

    I_mat = compute_epsilon_indicator_matrix(F_norm)
    fitness = compute_ibea_fitness(I_mat, kappa=kappa)

    # Penalize constraint violations so infeasible are deleted first
    fitness = fitness + 1e3 * CV_comb

    survivors = list(range(N_comb))

    while len(survivors) > target_size:
        # Find index with largest fitness (worst)
        sub_fit = [fitness[i] for i in survivors]
        worst_local_idx = int(np.argmax(sub_fit))
        worst_global_idx = survivors[worst_local_idx]

        del survivors[worst_local_idx]

        # Update remaining fitnesses: F(j) <- F(j) + exp(-I(worst, j) / kappa)
        for j in survivors:
            exp_term = np.exp(np.clip(-I_mat[worst_global_idx, j] / max(kappa, 1e-6), -50.0, 50.0))
            fitness[j] += exp_term

    survivors = np.array(survivors, dtype=int)
    return X_comb[survivors], F_comb[survivors], CV_comb[survivors]


def initialize(problem, pop_size=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return problem.random_solutions(pop_size, rng)


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, kappa=0.05, pc=0.9, pm=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # Tournament selection
    # Invert fitness for selection (lower F is better, so -F is higher for better)
    z_min = np.min(F, axis=0)
    z_max = np.max(F, axis=0)
    span = np.where(z_max - z_min <= 0, 1.0, z_max - z_min)
    Fn = (F - z_min) / span
    I_mat = compute_epsilon_indicator_matrix(Fn)
    fit = compute_ibea_fitness(I_mat, kappa=kappa) + 1e3 * CV

    mating_idx = np.empty(N, dtype=int)
    for i in range(N):
        a, b = rng.integers(N, size=2)
        mating_idx[i] = a if fit[a] <= fit[b] else b
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

    return environmental_selection(X_comb, F_comb, CV_comb, target_size=N, kappa=kappa)


def run(problem, pop_size=50, n_gen=50, kappa=0.05, pc=0.9, pm=None, seed=42):
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, kappa=kappa, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
