"""
Algorithm 37: GradNorm (Gradient Normalization for Adaptive Loss Balancing)
Proposed by: Chen, Badrinarayanan, Lee & Rabinovich (2018)
Family: Gradient-Based MOO (Part 5E, slides 38-40)

Core idea:
Dynamically re-weights per-task losses in multi-task learning so every task's
gradient norm trains at a comparable, self-correcting rate:

    1. Maintain trainable weights w_i per task loss (initially 1, sum w_i = k).
    2. At each step, measure:
       - Gradient norm: G_i(t) = ||∇_θ (w_i * L_i(t))|| = w_i * ||∇_θ L_i(t)||
       - Loss ratio: L_i(t) / L_i(0)
    3. Compute average gradient norm G_bar(t) and relative inverse training rate:
           r_i(t) = [L_i(t) / L_i(0)] / mean_j[L_j(t) / L_j(0)]
       (a slower-than-average task gets r_i > 1).
    4. Target gradient norm:
           target_i = G_bar(t) * (r_i(t))^alpha_asym
    5. GradNorm loss:
           L_grad = sum_i |G_i(t) - target_i|
       Update w_i via gradient descent against L_grad:
           w_i <- w_i - lr_w * sign(G_i - target_i) * g_i
       Renormalize: w_i <- w_i * (k / sum(w)).
    6. Update shared network weights θ via combined loss sum_i w_i * L_i.
"""

import numpy as np


def compute_gradnorm_targets(losses, initial_losses, grad_norms, alpha_asym=0.5):
    """
    Compute relative inverse training rates and GradNorm targets.
    """
    losses = np.asarray(losses, dtype=float)
    l0 = np.asarray(initial_losses, dtype=float)
    G = np.asarray(grad_norms, dtype=float)
    k = len(losses)

    G_bar = float(np.mean(G))
    ratios = losses / l0
    avg_ratio = float(np.mean(ratios))
    r = ratios / avg_ratio

    targets = G_bar * (r ** alpha_asym)
    return targets, G_bar, r


def update_task_weights(weights, base_grad_norms, targets, lr_w=0.025):
    """
    Update task weights w_i by descending L_grad = sum |w_i * g_i - target_i|
    and renormalizing sum w_i = k.
    """
    w = np.asarray(weights, dtype=float).copy()
    g = np.asarray(base_grad_norms, dtype=float)
    k = len(w)

    G = w * g
    diff = G - targets
    # d L_grad / d w_i = sign(G_i - target_i) * g_i
    grad_w = np.sign(diff) * g

    w_new = w - lr_w * grad_w
    # Renormalize to sum = k
    w_renorm = w_new * (k / np.sum(w_new))
    l_grad = float(np.sum(np.abs(diff)))
    return w_renorm, l_grad


def initialize(problem, initial_weights=None):
    k = problem.n_obj if hasattr(problem, "n_obj") else 2
    if initial_weights is None:
        weights = np.ones(k, dtype=float)
    else:
        weights = np.asarray(initial_weights, dtype=float)
    return weights


def evaluate(problem, theta):
    f, cv = problem.evaluate(theta)
    return (f[0] if f.ndim > 1 else f), (cv[0] if np.ndim(cv) > 0 else cv)


def step(problem, theta, weights, initial_losses, lr_theta=0.01, lr_w=0.025, alpha_asym=0.5):
    """
    One optimization step: update theta with weighted loss, then update weights w.
    """
    k = len(weights)
    losses, _ = evaluate(problem, theta)
    grads = problem.gradients(theta)  # (k, n_params)
    base_norms = np.linalg.norm(grads, axis=1)

    # 1. Update theta using weighted gradients
    total_grad = np.sum(weights[:, None] * grads, axis=0)
    theta_new = theta - lr_theta * total_grad

    # 2. Update task weights
    targets, _, _ = compute_gradnorm_targets(losses, initial_losses, weights * base_norms, alpha_asym=alpha_asym)
    weights_new, l_grad = update_task_weights(weights, base_norms, targets, lr_w=lr_w)

    return theta_new, weights_new, l_grad, losses


def run(problem, n_steps=50, lr_theta=0.01, lr_w=0.025, alpha_asym=0.5, seed=42):
    """
    Run GradNorm multi-task balancing loop.
    """
    rng = np.random.default_rng(seed)
    if hasattr(problem, "start"):
        theta = problem.start.copy()
    else:
        theta = rng.normal(0.0, 0.1, size=getattr(problem, "n_params", problem.n_var))

    k = problem.n_obj
    weights = np.ones(k, dtype=float)
    initial_losses, _ = evaluate(problem, theta)

    history_losses = [initial_losses.copy()]
    history_weights = [weights.copy()]

    for s in range(n_steps):
        theta, weights, l_grad, cur_losses = step(
            problem, theta, weights, initial_losses,
            lr_theta=lr_theta, lr_w=lr_w, alpha_asym=alpha_asym
        )
        history_losses.append(cur_losses.copy())
        history_weights.append(weights.copy())

    return theta, np.array(history_losses), np.array(history_weights)
