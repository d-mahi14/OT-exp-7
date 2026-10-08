"""
Algorithm 28: MO-TLBO (Multi-Objective Teaching-Learning-Based Optimization)
Proposed by: Rao & Patel (2013)
Family: Human-Behavior-Inspired (Part 4, slides 56-62)

Core idea:
Parameter-free algorithm simulating classroom teaching and peer learning:
1. Teacher Phase:
   - The teacher X_teacher is selected from the external Pareto archive by crowding
     distance (most isolated/least-crowded leader).
   - Class mean X_mean is computed across learners.
   - Each learner moves toward the teacher:
         X_new = X + r * (X_teacher - T_F * X_mean)
     where T_F in {1, 2} is the teaching factor.
   - Replace learner if X_new Pareto-dominates the current position.
2. Learner Phase:
   - Each learner interacts with a randomly chosen classmate j:
         X_new = X_i + r * (X_i - X_j) if i dominates j
         X_new = X_i + r * (X_j - X_i) if j dominates i
   - Replace learner if X_new Pareto-dominates the current position.
3. Archive stores all non-dominated solutions.
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


def select_teacher(archive_X, archive_F, rng):
    """
    Select teacher from archive favoring least-crowded solution.
    """
    if len(archive_X) == 1:
        return archive_X[0]
    cd = U.crowding_distance(archive_F)
    # Pick solution with largest crowding distance (or random among boundaries)
    max_cd = np.max(cd)
    candidates = np.where(cd == max_cd)[0]
    chosen = rng.choice(candidates)
    return archive_X[chosen]


def initialize(problem, pop_size=40, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pop_size, rng)
    F, CV = problem.evaluate(X)
    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), np.empty(0),
        X, F, CV, archive_size
    )
    return X, F, CV, arch_X, arch_F, arch_CV


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, arch_X, arch_F, arch_CV, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N, d = X.shape

    # 1. Teacher Phase
    teacher = select_teacher(arch_X, arch_F, rng)
    X_mean = np.mean(X, axis=0)

    for i in range(N):
        T_F = rng.integers(1, 3)  # 1 or 2
        r = rng.uniform(0.0, 1.0, size=d)
        X_new = X[i] + r * (teacher - T_F * X_mean)
        X_new = np.clip(X_new, problem.lower, problem.upper)
        if problem.var_type == "int":
            X_new = np.round(X_new)

        f_new, cv_new = problem.evaluate(X_new)
        f_new = f_new[0] if f_new.ndim > 1 else f_new
        cv_new = cv_new[0] if np.ndim(cv_new) > 0 else cv_new

        if U.constrained_dominates(f_new, cv_new, F[i], CV[i]):
            X[i] = X_new.copy()
            F[i] = f_new.copy()
            CV[i] = cv_new

        arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, X_new, f_new, cv_new, archive_size)

    # 2. Learner Phase
    for i in range(N):
        j = rng.integers(N - 1)
        if j >= i:
            j += 1

        r = rng.uniform(0.0, 1.0, size=d)
        if U.constrained_dominates(F[i], CV[i], F[j], CV[j]):
            X_new = X[i] + r * (X[i] - X[j])
        elif U.constrained_dominates(F[j], CV[j], F[i], CV[i]):
            X_new = X[i] + r * (X[j] - X[i])
        else:
            direction = 1.0 if rng.random() < 0.5 else -1.0
            X_new = X[i] + direction * r * (X[i] - X[j])

        X_new = np.clip(X_new, problem.lower, problem.upper)
        if problem.var_type == "int":
            X_new = np.round(X_new)

        f_new, cv_new = problem.evaluate(X_new)
        f_new = f_new[0] if f_new.ndim > 1 else f_new
        cv_new = cv_new[0] if np.ndim(cv_new) > 0 else cv_new

        if U.constrained_dominates(f_new, cv_new, F[i], CV[i]):
            X[i] = X_new.copy()
            F[i] = f_new.copy()
            CV[i] = cv_new

        arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, X_new, f_new, cv_new, archive_size)

    return X, F, CV, arch_X, arch_F, arch_CV


def run(problem, pop_size=40, archive_size=100, n_gen=50, seed=42):
    rng = np.random.default_rng(seed)
    X, F, CV, arch_X, arch_F, arch_CV = initialize(
        problem, pop_size=pop_size, archive_size=archive_size, rng=rng
    )

    for gen in range(n_gen):
        X, F, CV, arch_X, arch_F, arch_CV = step(
            problem, X, F, CV, arch_X, arch_F, arch_CV,
            archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
