"""
Algorithm 9: VIKOR
Full form: VlseKriterijumska Optimizacija I Kompromisno Resenje (Multicriteria Optimization and Compromise Solution)
Family: Ranking / MCDM (Part 1, slide 44; Part 5E, slides 32-34)

Core mechanism:
Rank alternatives by a compromise index Q that balances GROUP utility (average regret, S)
with INDIVIDUAL worst-case regret (R), via a tunable trade-off parameter v in [0, 1]:

    1. Determine best f_j* and worst f_j- per criterion (respecting min/max direction).
    2. Per alternative:
       Group utility S_i = sum_j w_j * (f_j* - f_ij) / (f_j* - f_j-)
       Individual regret R_i = max_j w_j * (f_j* - f_ij) / (f_j* - f_j-)
    3. S* = min S, S- = max S ; R* = min R, R- = max R
    4. Compromise index Q_i:
       Q_i = v * (S_i - S*) / (S- - S*) + (1 - v) * (R_i - R*) / (R- - R*)
    5. Rank ascending by Q_i (lowest Q_i is best).
    6. Check acceptable advantage Q(2nd) - Q(1st) >= 1 / (m - 1) and acceptable stability.
       Propose compromise solution or compromise set.

Default parameters:
- v = 0.5 (equal weight to group utility and individual regret)
"""

import numpy as np


def vikor(matrix, weights, v=0.5, maximize=None):
    """
    Perform VIKOR compromise ranking on an m x k decision matrix.

    Parameters
    ----------
    matrix : array-like, shape (m, k)
        Decision matrix of m alternatives and k criteria.
    weights : array-like, shape (k,)
        Criteria weights.
    v : float
        Trade-off parameter in [0, 1]. v > 0.5 emphasizes group utility;
        v < 0.5 emphasizes individual regret.
    maximize : array-like of bool, shape (k,), optional
        True for criteria to maximize, False for minimize.

    Returns
    -------
    S : ndarray, shape (m,)
        Group utility values.
    R : ndarray, shape (m,)
        Individual regret values.
    Q : ndarray, shape (m,)
        Compromise index values.
    rankings : ndarray, shape (m,)
        Rank order (0-indexed, where rankings[0] is the index of the top alternative).
    compromise_set : list of int
        Indices of alternatives that form the compromise solution/set.
    """
    X = np.asarray(matrix, dtype=float)
    m, k = X.shape
    w = np.asarray(weights, dtype=float)

    if maximize is None:
        maximize = np.zeros(k, dtype=bool)
    else:
        maximize = np.asarray(maximize, dtype=bool)

    if not np.isclose(np.sum(w), 1.0):
        w = w / np.sum(w)

    # 1. Best f* and worst f- per criterion
    f_star = np.zeros(k)
    f_minus = np.zeros(k)

    for j in range(k):
        if maximize[j]:
            f_star[j] = np.max(X[:, j])
            f_minus[j] = np.min(X[:, j])
        else:
            f_star[j] = np.min(X[:, j])
            f_minus[j] = np.max(X[:, j])

    # 2. Compute S_i and R_i
    S = np.zeros(m)
    R = np.zeros(m)

    for i in range(m):
        terms = np.zeros(k)
        for j in range(k):
            denom = f_star[j] - f_minus[j] if maximize[j] else f_minus[j] - f_star[j]
            if np.isclose(denom, 0.0):
                terms[j] = 0.0
            else:
                num = (f_star[j] - X[i, j]) if maximize[j] else (X[i, j] - f_star[j])
                terms[j] = w[j] * (num / denom)
        S[i] = np.sum(terms)
        R[i] = np.max(terms)

    # 3. Min/Max of S and R
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    # 4. Compute Q_i
    Q = np.zeros(m)
    denom_S = S_minus - S_star if not np.isclose(S_minus, S_star) else 1e-12
    denom_R = R_minus - R_star if not np.isclose(R_minus, R_star) else 1e-12

    for i in range(m):
        Q[i] = v * (S[i] - S_star) / denom_S + (1.0 - v) * (R[i] - R_star) / denom_R

    # 5. Rank by Q ascending
    rankings = np.argsort(Q)

    # 6. Acceptable advantage and stability checks
    DQ = 1.0 / (m - 1) if m > 1 else 1.0
    best_idx = rankings[0]
    second_idx = rankings[1] if m > 1 else rankings[0]

    cond1 = (Q[second_idx] - Q[best_idx]) >= DQ - 1e-9

    # Condition 2: best in Q is also best in S or R
    cond2 = (best_idx == np.argmin(S)) or (best_idx == np.argmin(R))

    compromise_set = [best_idx]
    if cond1 and cond2:
        pass  # best_idx is the unique compromise winner
    elif not cond1:
        # Include all alternatives where Q(i) - Q(1st) < DQ
        for idx in rankings[1:]:
            if Q[idx] - Q[best_idx] < DQ - 1e-9:
                compromise_set.append(idx)
            else:
                break
    elif not cond2:
        # Both first and second are proposed
        if second_idx not in compromise_set:
            compromise_set.append(second_idx)

    return S, R, Q, rankings, compromise_set


def initialize(problem_or_matrix, weights=None, v=0.5, maximize=None):
    """
    Initialize decision matrix, weights, and parameters.
    """
    if hasattr(problem_or_matrix, "matrix"):
        matrix = problem_or_matrix.matrix
        if weights is None and hasattr(problem_or_matrix, "weights"):
            weights = problem_or_matrix.weights
        if maximize is None and hasattr(problem_or_matrix, "maximize"):
            maximize = problem_or_matrix.maximize
    else:
        matrix = np.asarray(problem_or_matrix, dtype=float)

    k = matrix.shape[1]
    if weights is None:
        weights = np.ones(k) / k
    if maximize is None:
        maximize = np.zeros(k, dtype=bool)

    return matrix, weights, v, maximize


def evaluate(matrix, weights, v=0.5, maximize=None):
    """
    Evaluate VIKOR metrics.
    """
    return vikor(matrix, weights, v, maximize)


def step(matrix, weights, v=0.5, maximize=None):
    """
    Perform one ranking step.
    """
    return vikor(matrix, weights, v, maximize)


def run(problem_or_matrix, weights=None, v=0.5, maximize=None, candidates=None):
    """
    Run VIKOR on a problem or decision matrix.
    Returns:
        rankings : sorted indices of alternatives (best to worst)
        Q : compromise indices
        compromise_set : indices of compromise solutions
    """
    if hasattr(problem_or_matrix, "matrix"):
        mat, w, v_val, mx = initialize(problem_or_matrix, weights, v, maximize)
    elif hasattr(problem_or_matrix, "evaluate") and candidates is not None:
        F, _ = problem_or_matrix.evaluate(candidates)
        mat = problem_or_matrix.to_natural(F) if hasattr(problem_or_matrix, "to_natural") else F
        mx = getattr(problem_or_matrix, "maximize", np.zeros(mat.shape[1], bool))
        w = np.ones(mat.shape[1]) / mat.shape[1] if weights is None else weights
        v_val = v
    else:
        mat, w, v_val, mx = initialize(problem_or_matrix, weights, v, maximize)

    S, R, Q, rankings, comp_set = vikor(mat, w, v_val, mx)
    return rankings, Q, comp_set
