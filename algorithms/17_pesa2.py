"""
Algorithm 17: PESA-II (Pareto Envelope-Based Selection Algorithm II)
Proposed by: Corne, Jerram, Knowles & Oates (2001)
Family: Elitist, Grid-Based MOEA (Part 3, slides 28-35)
Task C Deep-Dive Algorithm for Roll No. 17.

Core idea:
Maintains diversity using a hyper-grid partition over objective space (the Pareto envelope).
Instead of selecting individuals directly by individual fitness or crowding distance,
PESA-II performs REGION-BASED SELECTION:

    1. Divide objective space into hyperboxes (a regular grid with G divisions per objective).
    2. Place each non-dominated solution in the archive into its corresponding grid cell.
    3. Count the number of solutions per occupied cell (the squeeze factor):
           squeeze(cell) = count(solutions in cell)
    4. Region-based selection:
       Run a tournament between occupied grid cells, choosing the cell with the
       LOWEST squeeze factor (least crowded). Then pick one individual uniformly
       at random from that chosen cell to become a parent!
    5. Environmental selection:
       Only non-dominated solutions enter the archive. If the archive overflows
       its capacity, remove an individual from the most crowded cell (highest squeeze factor).
    6. Variation (crossover + mutation) creates the offspring population.

Pros: Very fast O(kN) selection; scales well to many objectives; intuitive hyper-grid.
Cons: Grid resolution G needs tuning; sensitive to boundary changes; fixed grid can distort irregular Pareto fronts.
Complexity: O(k N) per generation.
"""

import numpy as np
import moo_utils as U
import operators as ops


def compute_grid_indices(F, n_divs=10, f_min=None, f_max=None):
    """
    Map each objective vector to its grid cell coordinates.

    Parameters
    ----------
    F : array-like, shape (N, k)
        Objective vectors.
    n_divs : int
        Number of divisions per objective axis.
    f_min, f_max : array-like, shape (k,), optional
        Grid bounding coordinates.

    Returns
    -------
    cell_tuples : list of tuple
        Grid coordinate tuple for each individual.
    grid_coords : ndarray, shape (N, k)
        Integer grid coordinates.
    """
    F = np.asarray(F, dtype=float)
    N, k = F.shape
    if f_min is None:
        f_min = F.min(axis=0)
    if f_max is None:
        f_max = F.max(axis=0)

    span = f_max - f_min
    span = np.where(span <= 0, 1.0, span)

    norm_F = (F - f_min) / span
    # Grid coordinate: int in [0, n_divs - 1]
    grid_coords = np.clip(np.floor(norm_F * n_divs).astype(int), 0, n_divs - 1)
    cell_tuples = [tuple(row) for row in grid_coords]
    return cell_tuples, grid_coords


def compute_squeeze_factors(cell_tuples):
    """
    Compute squeeze factor for each unique occupied cell and map cells to member indices.
    """
    cell_members = {}
    for idx, cell in enumerate(cell_tuples):
        cell_members.setdefault(cell, []).append(idx)

    squeeze = {cell: len(members) for cell, members in cell_members.items()}
    return squeeze, cell_members


def region_based_selection(cell_members, squeeze, n_select, rng):
    """
    PESA-II signature selection rule:
    Tournament between occupied grid cells favoring lower squeeze factor.
    Then select one individual uniformly at random from the winning cell.
    """
    cells = list(cell_members.keys())
    selected_indices = np.empty(n_select, dtype=int)

    for i in range(n_select):
        # Binary tournament between two randomly drawn cells
        if len(cells) == 1:
            chosen_cell = cells[0]
        else:
            c1_idx, c2_idx = rng.integers(len(cells), size=2)
            c1, c2 = cells[c1_idx], cells[c2_idx]
            if squeeze[c1] < squeeze[c2]:
                chosen_cell = c1
            elif squeeze[c2] < squeeze[c1]:
                chosen_cell = c2
            else:
                chosen_cell = c1 if rng.random() < 0.5 else c2

        # Pick random individual from the chosen cell
        members = cell_members[chosen_cell]
        selected_indices[i] = members[rng.integers(len(members))]

    return selected_indices


def update_archive(archive_X, archive_F, archive_CV, new_X, new_F, new_CV, max_size, n_divs=10, rng=None):
    """
    Update archive with non-dominated solutions and prune if capacity exceeded.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if len(archive_X) == 0:
        comb_X = new_X
        comb_F = new_F
        comb_CV = new_CV
    else:
        comb_X = np.vstack([archive_X, new_X])
        comb_F = np.vstack([archive_F, new_F])
        comb_CV = np.concatenate([archive_CV, new_CV])

    # Filter to non-dominated individuals
    nd_idx = U.pareto_front(comb_F, comb_CV)
    arch_X = comb_X[nd_idx]
    arch_F = comb_F[nd_idx]
    arch_CV = comb_CV[nd_idx]

    # If archive size exceeds max_size, prune from cell with highest squeeze factor
    while len(arch_X) > max_size:
        cell_tuples, _ = compute_grid_indices(arch_F, n_divs=n_divs)
        squeeze, cell_members = compute_squeeze_factors(cell_tuples)

        # Find cell with largest squeeze factor (> 1)
        max_sq = max(squeeze.values())
        crowded_cells = [c for c, sq in squeeze.items() if sq == max_sq]
        chosen_crowded = crowded_cells[rng.integers(len(crowded_cells))]

        # Remove random individual from that crowded cell
        members = cell_members[chosen_crowded]
        remove_idx = members[rng.integers(len(members))]

        arch_X = np.delete(arch_X, remove_idx, axis=0)
        arch_F = np.delete(arch_F, remove_idx, axis=0)
        arch_CV = np.delete(arch_CV, remove_idx, axis=0)

    return arch_X, arch_F, arch_CV


def initialize(problem, pop_size=100, archive_size=100, rng=None):
    """
    Initialize population and empty archive.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pop_size, rng)
    return X


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, archive_X, archive_F, archive_CV, archive_size=100, n_divs=10, pc=0.9, pm=None, rng=None):
    """
    Perform one PESA-II generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # 1. Update archive with current population
    archive_X, archive_F, archive_CV = update_archive(
        archive_X, archive_F, archive_CV, X, F, CV, max_size=archive_size, n_divs=n_divs, rng=rng
    )

    # 2. Region-based selection from archive to form mating pool
    cell_tuples, _ = compute_grid_indices(archive_F, n_divs=n_divs)
    squeeze, cell_members = compute_squeeze_factors(cell_tuples)
    parent_idx = region_based_selection(cell_members, squeeze, N, rng)
    parents = archive_X[parent_idx]

    # 3. Create offspring via crossover and mutation
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

    return X_next, F_next, CV_next, archive_X, archive_F, archive_CV


def run(problem, pop_size=100, archive_size=100, n_gen=50, n_divs=10, pc=0.9, pm=None, seed=42):
    """
    Run PESA-II optimization.
    """
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, archive_size=archive_size, rng=rng)
    F, CV = evaluate(problem, X)

    archive_X = np.empty((0, problem.n_var))
    archive_F = np.empty((0, problem.n_obj))
    archive_CV = np.empty(0)

    for gen in range(n_gen):
        X, F, CV, archive_X, archive_F, archive_CV = step(
            problem, X, F, CV, archive_X, archive_F, archive_CV,
            archive_size=archive_size, n_divs=n_divs, pc=pc, pm=pm, rng=rng
        )

    # Final archive update with last generation
    archive_X, archive_F, _ = update_archive(
        archive_X, archive_F, archive_CV, X, F, CV, max_size=archive_size, n_divs=n_divs, rng=rng
    )
    return archive_X, archive_F
