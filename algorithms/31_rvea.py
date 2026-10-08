"""
Algorithm 31: RVEA (Reference Vector Guided Evolutionary Algorithm)
Proposed by: Cheng, Olhofer & Jin (2016)
Family: Many-Objective MOEA (Part 5E, slides 8-10)

Core idea:
Uses unit reference vectors v_1, ..., v_H and an Angle-Penalized Distance (APD)
that dynamically balances convergence and diversity as generations progress:

    1. Generate unit reference vectors v_1, ..., v_H on the k-D simplex (Das-Dennis).
    2. Normalize objectives every generation using ideal point z* and nadir estimate.
    3. Partition population: assign each individual to the reference vector with
       the smallest acute angle theta.
    4. Compute Angle-Penalized Distance (APD):
           APD(x) = d(x) * [ 1 + alpha_pen * (t / t_max)^alpha_pen * (theta(x) / gamma_min) ]
       where d(x) = ||f_norm(x)||, alpha_pen = 2 (fixed penalty strength),
       and gamma_min is the smallest angle between neighboring reference vectors.
    5. Within each sector, keep only the individual with the smallest APD (one survivor per sector).
    6. Generate offspring and repeat.
"""

import numpy as np
import moo_utils as U
import operators as ops


def generate_unit_reference_vectors(k, p=4):
    """
    Generate unit reference vectors on k-D simplex.
    """
    W = U.das_dennis(k, p)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    V = W / norms
    return V


def compute_gamma_min(V):
    """
    Compute smallest angle between any two distinct reference vectors.
    """
    H = len(V)
    if H <= 1:
        return 1.0
    cosine = np.clip(V @ V.T, -1.0, 1.0)
    # Exclude diagonal (cos=1)
    np.fill_diagonal(cosine, -1.0)
    max_cos = np.max(cosine)
    gamma_min = np.arccos(np.clip(max_cos, -1.0, 1.0))
    return max(gamma_min, 1e-4)


def assign_and_compute_apd(F_norm, V, gamma_min, t, t_max, alpha_pen=2.0):
    """
    Assign each solution to nearest reference vector and calculate APD.
    """
    N = len(F_norm)
    H = len(V)

    d_norm = np.linalg.norm(F_norm, axis=1)
    d_norm_safe = np.where(d_norm == 0, 1e-6, d_norm)
    F_unit = F_norm / d_norm_safe[:, None]

    # Cosine of angle to each reference vector
    cos_theta = np.clip(F_unit @ V.T, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # shape (N, H)

    # Nearest vector for each solution
    partitions = np.argmin(theta, axis=1)

    # Penalty factor P(t)
    p_t = alpha_pen * ((t / max(1, t_max)) ** alpha_pen)

    apd = np.zeros(N)
    for i in range(N):
        sec = partitions[i]
        angle = theta[i, sec]
        apd[i] = d_norm[i] * (1.0 + p_t * (angle / gamma_min))

    return partitions, apd


def rvea_selection(X_comb, F_comb, CV_comb, V, gamma_min, t, t_max, target_size, alpha_pen=2.0):
    """
    RVEA sector-exclusive survivor selection.
    """
    z_ideal = np.min(F_comb, axis=0)
    z_nadir = np.max(F_comb, axis=0)
    span = z_nadir - z_ideal
    span = np.where(span <= 0, 1.0, span)
    F_norm = (F_comb - z_ideal) / span

    partitions, apd = assign_and_compute_apd(F_norm, V, gamma_min, t, t_max, alpha_pen=alpha_pen)

    # Add huge penalty to apd for constraint violation
    apd_penalized = apd + 1e6 * CV_comb

    survivors = []
    H = len(V)

    # In each sector, keep the individual with smallest APD
    for sec in range(H):
        members = np.where(partitions == sec)[0]
        if len(members) > 0:
            best_in_sec = members[np.argmin(apd_penalized[members])]
            survivors.append(best_in_sec)

    # If survivors fewer than target_size, backfill by best overall APD
    if len(survivors) < target_size:
        remaining = [i for i in range(len(X_comb)) if i not in survivors]
        sorted_rem = sorted(remaining, key=lambda i: apd_penalized[i])
        survivors.extend(sorted_rem[:target_size - len(survivors)])
    elif len(survivors) > target_size:
        # Sort survivors by APD and truncate
        survivors = sorted(survivors, key=lambda i: apd_penalized[i])[:target_size]

    survivors = np.array(survivors, dtype=int)
    return X_comb[survivors], F_comb[survivors], CV_comb[survivors]


def initialize(problem, p=4, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    V = generate_unit_reference_vectors(problem.n_obj, p=p)
    H = len(V)
    N = max(H, 12)
    X = problem.random_solutions(N, rng)
    return V, X


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, V, gamma_min, t, t_max, pc=0.9, pm=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # Offspring creation
    offspring = []
    for i in range(0, N, 2):
        p1 = X[i]
        p2 = X[(i + 1) % N]
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

    return rvea_selection(X_comb, F_comb, CV_comb, V, gamma_min, t, t_max, target_size=N)


def run(problem, p=4, n_gen=50, pc=0.9, pm=None, seed=42):
    rng = np.random.default_rng(seed)
    V, X = initialize(problem, p=p, rng=rng)
    F, CV = evaluate(problem, X)
    gamma_min = compute_gamma_min(V)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, V, gamma_min, gen, n_gen, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
