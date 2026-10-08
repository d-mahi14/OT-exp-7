"""
Algorithm 1: Weighted Sum Method

Multiobjective optimization is converted into a single-objective
optimization problem using a weighted sum of objectives.

All objectives are assumed to be minimization objectives.

Lecture idea:
    min F(x) = w1*f1(x) + w2*f2(x) + ... + wm*fm(x)

where:
    wi >= 0
    sum(wi) = 1
"""

import numpy as np


def weighted_sum(objectives, weights):
    """
    Calculate the weighted-sum scalar fitness.

    Parameters
    ----------
    objectives : array-like
        Objective values [f1, f2, ..., fm].

    weights : array-like
        Non-negative weights [w1, w2, ..., wm].
        They should sum to 1.

    Returns
    -------
    float
        Weighted-sum objective value.
    """

    objectives = np.asarray(objectives, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if objectives.ndim != 1:
        raise ValueError("objectives must be a 1-D array")

    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")

    if len(objectives) != len(weights):
        raise ValueError(
            "objectives and weights must have the same length"
        )

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("weights must sum to 1")

    return float(np.dot(objectives, weights))


def solve_weighted_sum(problem, weights, candidates):
    """
    Select the best candidate using the weighted-sum method.

    Parameters
    ----------
    problem : problem object
        Problem containing an evaluate(x) method.

    weights : array-like
        Objective weights.

    candidates : array-like
        Candidate decision vectors.

    Returns
    -------
    best_x : ndarray
        Decision vector of the best candidate.

    best_f : ndarray
        Objective values of the best candidate.

    best_value : float
        Weighted-sum scalar value.
    """

    best_x = None
    best_f = None
    best_value = np.inf

    for x in candidates:
        f = np.asarray(problem.evaluate(x), dtype=float)

        value = weighted_sum(f, weights)

        if value < best_value:
            best_value = value
            best_x = np.asarray(x).copy()
            best_f = f.copy()

    return best_x, best_f, best_value


if __name__ == "__main__":
    # Small standalone demonstration

    f = np.array([2.0, 5.0])
    w = np.array([0.4, 0.6])

    result = weighted_sum(f, w)

    print("Objectives:", f)
    print("Weights:", w)
    print("Weighted-sum value:", result)