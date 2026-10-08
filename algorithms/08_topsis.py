"""
Algorithm 8: TOPSIS
Full form: Technique for Order of Preference by Similarity to Ideal Solution
Family: Ranking / MCDM (Part 1, slide 43; Part 5E, slides 29-31)

Core mechanism:
Rank a GIVEN discrete set of m alternatives across k criteria by how close
each is to an ideal solution (A+) and how far from an anti-ideal solution (A-),
combined into a single closeness coefficient C_i:

    1. Vector-normalize: r_ij = x_ij / sqrt(sum_i x_ij^2)
    2. Apply weights: v_ij = w_j * r_ij
    3. Determine ideal A+ (best per criterion) and anti-ideal A- (worst per criterion)
    4. Compute Euclidean distances d_i+ = ||v_i - A+||, d_i- = ||v_i - A-||
    5. Closeness coefficient: C_i = d_i- / (d_i+ + d_i-)
    6. Rank descending by C_i (highest C_i is best)

Note:
TOPSIS is a post-optimization MCDM ranking tool, not a generator of solutions.
It ranks an existing Pareto front or a discrete candidate matrix (e.g. P12).
"""

import numpy as np


def vector_normalize(matrix):
    """
    Column-wise Euclidean vector normalization:
    r_ij = x_ij / sqrt(sum_i x_ij^2)
    """
    X = np.asarray(matrix, dtype=float)
    col_norms = np.linalg.norm(X, axis=0)
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    return X / col_norms


def topsis(matrix, weights, maximize=None):
    """
    Perform TOPSIS ranking on an m x k decision matrix.

    Parameters
    ----------
    matrix : array-like, shape (m, k)
        Decision matrix of m alternatives and k criteria.
    weights : array-like, shape (k,)
        Criteria weights summing to 1.
    maximize : array-like of bool, shape (k,), optional
        True for criteria to maximize, False for minimize.
        Default is all False (minimization).

    Returns
    -------
    closeness : ndarray, shape (m,)
        Closeness coefficients C_i in [0, 1].
    rankings : ndarray, shape (m,)
        Rank order (0-indexed, where rankings[0] is the index of the top alternative).
    d_plus : ndarray, shape (m,)
        Distances to the ideal solution A+.
    d_minus : ndarray, shape (m,)
        Distances to the anti-ideal solution A-.
    A_plus : ndarray, shape (k,)
        Ideal solution in weighted normalized space.
    A_minus : ndarray, shape (k,)
        Anti-ideal solution in weighted normalized space.
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

    # 1. Vector normalization
    R = vector_normalize(X)

    # 2. Weighted normalized matrix
    V = R * w

    # 3. Determine ideal (A+) and anti-ideal (A-)
    A_plus = np.zeros(k)
    A_minus = np.zeros(k)

    for j in range(k):
        if maximize[j]:
            A_plus[j] = np.max(V[:, j])
            A_minus[j] = np.min(V[:, j])
        else:
            A_plus[j] = np.min(V[:, j])
            A_minus[j] = np.max(V[:, j])

    # 4. Euclidean distances
    d_plus = np.linalg.norm(V - A_plus, axis=1)
    d_minus = np.linalg.norm(V - A_minus, axis=1)

    # 5. Closeness coefficient
    denom = d_plus + d_minus
    denom = np.where(denom == 0, 1e-12, denom)
    C = d_minus / denom

    # 6. Rank descending by C
    rankings = np.argsort(-C)

    return C, rankings, d_plus, d_minus, A_plus, A_minus


def initialize(problem_or_matrix, weights=None, maximize=None):
    """
    Initialize decision matrix, weights, and criterion directions.
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

    return matrix, weights, maximize


def evaluate(matrix, weights, maximize=None):
    """
    Evaluate closeness scores and rankings.
    """
    return topsis(matrix, weights, maximize)


def step(matrix, weights, maximize=None):
    """
    Perform one ranking step.
    """
    return topsis(matrix, weights, maximize)


def run(problem_or_matrix, weights=None, maximize=None, candidates=None):
    """
    Run TOPSIS on a problem or decision matrix.
    If a continuous Problem object is passed with candidates, evaluates candidates first.
    Returns:
        rankings : indices of alternatives from best to worst
        C : closeness coefficients
    """
    if hasattr(problem_or_matrix, "matrix"):
        mat, w, mx = initialize(problem_or_matrix, weights, maximize)
    elif hasattr(problem_or_matrix, "evaluate") and candidates is not None:
        F, _ = problem_or_matrix.evaluate(candidates)
        mat = problem_or_matrix.to_natural(F) if hasattr(problem_or_matrix, "to_natural") else F
        mx = getattr(problem_or_matrix, "maximize", np.zeros(mat.shape[1], bool))
        w = np.ones(mat.shape[1]) / mat.shape[1] if weights is None else weights
    else:
        mat, w, mx = initialize(problem_or_matrix, weights, maximize)

    C, rankings, _, _, _, _ = topsis(mat, w, mx)
    return rankings, C
