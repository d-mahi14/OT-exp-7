"""
Algorithm 36: MGDA (Multiple-Gradient Descent Algorithm)
Proposed by: Désidéri (2012)
Family: Gradient-Based MOO (Part 5E, slides 35-37)

Core idea:
For differentiable multiobjective problems (e.g. P13 Multi-task learning):
Find the minimum-norm vector in the convex hull of the per-objective gradients.
The resulting direction d* is a common descent direction that weakly decreases
every objective simultaneously:

    1. At current point x, compute gradients g_i = ∇f_i(x) for i = 1, ..., k.
    2. Solve for minimum-norm point d* in the convex hull conv(g_1, ..., g_k):
           min || sum_{i=1}^k alpha_i * g_i ||^2
           subject to sum alpha_i = 1, alpha_i >= 0
       For k = 2, this has closed-form solution:
           alpha* = clip( ((g_2 - g_1) . g_2) / ||g_1 - g_2||^2, 0.0, 1.0 )
           d* = alpha* * g_1 + (1 - alpha*) * g_2
    3. If ||d*|| < tol: x is Pareto-stationary (cannot improve any objective without harming another).
    4. Otherwise update x <- x - eta * d*.
"""

import numpy as np


def find_min_norm_element_2d(g1, g2):
    """
    Closed-form solution for minimum norm vector in convex hull of two gradient vectors.
    min || alpha * g1 + (1 - alpha) * g2 ||^2  s.t. 0 <= alpha <= 1.
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)

    diff = g1 - g2
    diff_norm_sq = float(np.dot(diff, diff))

    if diff_norm_sq < 1e-12:
        return 0.5, g1.copy()

    # Closed form: ((g2 - g1) . g2) / ||g1 - g2||^2 = - (diff . g2) / diff_norm_sq
    alpha = float(np.dot(g2 - g1, g2) / diff_norm_sq)
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    d_star = alpha_clamped * g1 + (1.0 - alpha_clamped) * g2
    return alpha_clamped, d_star


def find_min_norm_element_general(grads):
    """
    Find minimum norm vector in convex hull of k gradients using Frank-Wolfe / QP.
    """
    grads = np.asarray(grads, dtype=float)
    k, d = grads.shape
    if k == 1:
        return np.array([1.0]), grads[0]
    if k == 2:
        a, d_star = find_min_norm_element_2d(grads[0], grads[1])
        return np.array([a, 1.0 - a]), d_star

    # Projected gradient descent on simplex
    alpha = np.ones(k) / k
    for _ in range(100):
        d_vec = alpha @ grads
        grad_alpha = 2.0 * (grads @ d_vec)
        # Frank-Wolfe step
        best_corner = np.argmin(grad_alpha)
        gamma = 2.0 / (2.0 + _)
        alpha = (1.0 - gamma) * alpha
        alpha[best_corner] += gamma

    d_star = alpha @ grads
    return alpha, d_star


def initialize(problem, start_point=None):
    if start_point is not None:
        x = np.array(start_point, dtype=float)
    elif hasattr(problem, "start"):
        x = np.array(problem.start, dtype=float)
    else:
        x = (problem.lower + problem.upper) / 2.0
    return x


def evaluate(problem, x):
    f, cv = problem.evaluate(x)
    return (f[0] if f.ndim > 1 else f), (cv[0] if np.ndim(cv) > 0 else cv)


def step(problem, x, eta=0.1, tol=1e-5):
    """
    Perform one MGDA step.
    """
    grads = problem.gradients(x)  # shape (k, d)
    alpha, d_star = find_min_norm_element_general(grads)
    norm_d = float(np.linalg.norm(d_star))

    if norm_d < tol:
        return x, alpha, d_star, True

    x_new = x - eta * d_star
    if hasattr(problem, "lower") and hasattr(problem, "upper"):
        x_new = np.clip(x_new, problem.lower, problem.upper)

    return x_new, alpha, d_star, False


def run(problem, start_point=None, eta=0.1, max_steps=100, tol=1e-5):
    """
    Run MGDA from start_point.
    Returns:
        trajectory_x : list of points
        trajectory_f : list of objective vectors
    """
    x = initialize(problem, start_point)
    traj_x = [x.copy()]
    f, _ = evaluate(problem, x)
    traj_f = [f.copy()]

    for step_i in range(max_steps):
        x_next, alpha, d_star, stationary = step(problem, x, eta=eta, tol=tol)
        if stationary:
            break
        x = x_next
        traj_x.append(x.copy())
        f, _ = evaluate(problem, x)
        traj_f.append(f.copy())

    return np.array(traj_x), np.array(traj_f)
