"""
Algorithm 29: MOBA (Multi-Objective Bat Algorithm)
Proposed by: Yang (2011), extended to MOO
Family: Swarm (Part 4, slides 63-69)
Task C Deep-Dive Algorithm for Roll No. 17.

Core idea:
Each bat's frequency-driven velocity is pulled toward (or for diversity, steered relative to)
an archive leader chosen by crowding distance. Loudness decays over time, shifting the
swarm from exploration to exploitation:

    f_i = f_min + (f_max - f_min) * beta,  beta in [0, 1]
    v_i = v_i + (x_best - x_i) * f_i
    x_new = x_i + v_i
    A_{t+1} = alpha * A_t

Local random walk when rand > r_i (pulse emission rate):
    x_new = x_best + epsilon * A_mean
    r_{t+1} = r_0 * (1 - exp(-gamma * t))

Archive maintenance:
- An external archive stores all non-dominated solutions discovered.
- x_best is drawn from the archive favoring less-crowded regions (large crowding distance).
- Archive is pruned by crowding distance when capacity is exceeded.

Default parameters:
- f_min = 0.0, f_max = 2.0 (frequency range)
- A_0 = 1.0, alpha = 0.9 (loudness decay)
- r_0 = 0.5, gamma = 0.05 (pulse rate schedule)
"""

import numpy as np
import moo_utils as U


def update_archive(archive_X, archive_F, archive_CV, new_X, new_F, new_CV, max_size=100):
    if len(archive_X) == 0:
        comb_X = np.atleast_2d(new_X)
        comb_F = np.atleast_2d(new_F)
        comb_CV = np.atleast_1d(new_CV)
    else:
        comb_X = np.vstack([archive_X, np.atleast_2d(new_X)])
        comb_F = np.vstack([archive_F, np.atleast_2d(new_F)])
        comb_CV = np.concatenate([archive_CV, np.atleast_1d(new_CV)])

    nd_idx = U.pareto_front(comb_F, comb_CV)
    arch_X = comb_X[nd_idx]
    arch_F = comb_F[nd_idx]
    arch_CV = comb_CV[nd_idx]

    while len(arch_X) > max_size:
        cd = U.crowding_distance(arch_F)
        worst_idx = np.argmin(cd)
        arch_X = np.delete(arch_X, worst_idx, axis=0)
        arch_F = np.delete(arch_F, worst_idx, axis=0)
        arch_CV = np.delete(arch_CV, worst_idx, axis=0)

    return arch_X, arch_F, arch_CV


def select_leader(archive_X, archive_F, rng):
    """
    Select x_best leader from archive favoring solutions with larger crowding distance.
    """
    if len(archive_X) == 1:
        return archive_X[0]

    cd = U.crowding_distance(archive_F)
    finite_mask = ~np.isinf(cd)
    max_finite = np.max(cd[finite_mask]) if np.any(finite_mask) else 1.0
    weights = np.where(np.isinf(cd), 2.0 * max_finite, cd)
    weights = np.maximum(weights, 1e-6)
    probs = weights / np.sum(weights)

    idx = rng.choice(len(archive_X), p=probs)
    return archive_X[idx]


def initialize(problem, n_bats=40, archive_size=100, f_min=0.0, f_max=2.0, A0=1.0, r0=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(n_bats, rng)
    V = np.zeros_like(X)
    F, CV = problem.evaluate(X)

    freq = np.full(n_bats, f_min)
    loudness = np.full(n_bats, A0)
    pulse_rate = np.full(n_bats, r0)

    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), np.empty(0),
        X, F, CV, archive_size
    )
    return X, V, F, CV, freq, loudness, pulse_rate, arch_X, arch_F, arch_CV


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, V, F, CV, freq, loudness, pulse_rate, arch_X, arch_F, arch_CV,
         f_min=0.0, f_max=2.0, alpha=0.9, gamma=0.05, t=0, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N, d = X.shape

    A_mean = float(np.mean(loudness))

    for i in range(N):
        x_best = select_leader(arch_X, arch_F, rng)

        # 1. Frequency update
        beta = rng.uniform(0.0, 1.0)
        freq[i] = f_min + (f_max - f_min) * beta

        # 2. Velocity and position update
        V[i] = V[i] + (x_best - X[i]) * freq[i]
        x_new = X[i] + V[i]

        # 3. Local random walk around best if rand > pulse_rate
        if rng.random() > pulse_rate[i]:
            eps = rng.uniform(-1.0, 1.0, size=d)
            x_new = x_best + eps * A_mean * (problem.upper - problem.lower) * 0.05

        x_new = np.clip(x_new, problem.lower, problem.upper)
        if problem.var_type == "int":
            x_new = np.round(x_new)

        f_new, cv_new = problem.evaluate(x_new)
        f_new = f_new[0] if f_new.ndim > 1 else f_new
        cv_new = cv_new[0] if np.ndim(cv_new) > 0 else cv_new

        # 4. Acceptance: if dominates or rand < loudness
        if U.constrained_dominates(f_new, cv_new, F[i], CV[i]) or (rng.random() < loudness[i] and not U.constrained_dominates(F[i], CV[i], f_new, cv_new)):
            X[i] = x_new.copy()
            F[i] = f_new.copy()
            CV[i] = cv_new
            loudness[i] *= alpha
            pulse_rate[i] = 1.0 - np.exp(-gamma * (t + 1))

        # Update archive
        arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, x_new, f_new, cv_new, archive_size)

    return X, V, F, CV, freq, loudness, pulse_rate, arch_X, arch_F, arch_CV


def run(problem, n_bats=40, archive_size=100, n_gen=50, f_min=0.0, f_max=2.0, alpha=0.9, gamma=0.05, seed=42):
    rng = np.random.default_rng(seed)
    (X, V, F, CV, freq, loudness, pulse_rate,
     arch_X, arch_F, arch_CV) = initialize(
        problem, n_bats=n_bats, archive_size=archive_size,
        f_min=f_min, f_max=f_max, rng=rng
    )

    for gen in range(n_gen):
        (X, V, F, CV, freq, loudness, pulse_rate,
         arch_X, arch_F, arch_CV) = step(
            problem, X, V, F, CV, freq, loudness, pulse_rate,
            arch_X, arch_F, arch_CV,
            f_min=f_min, f_max=f_max, alpha=alpha, gamma=gamma,
            t=gen, archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
