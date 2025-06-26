import numpy as np
import random


def local_search(objective_function, initial_solution, generate_neighbors,
                 nc, nf, meters, lines, wires_per_feeder,
                 max_iterations=1000, verbose=False):
    """
    Perform local search to optimize the objective function.

    Parameters:
        objective_function: function(solution) -> float
        initial_solution: numpy array
        generate_neighbors: function(solution) -> list of numpy arrays
        max_iterations: int
        verbose: bool

    Returns:
        best_solution: numpy array
        best_score: float
    """
    current_solution = initial_solution.copy()
    current_score = objective_function(
        solution = current_solution,
        nc = nc,
        nf = nf,
        meters = meters,
        lines = lines,
        wires_per_feeder = wires_per_feeder
    )

    for iteration in range(max_iterations):
        neighbors = generate_neighbors(current_solution)
        if not neighbors:
            break  # No neighbors to explore

        scores = [objective_function(solution=n, nc=nc, nf=nf, meters=meters, lines=lines, wires_per_feeder=wires_per_feeder) for n in neighbors]
        best_neighbor_idx = np.argmin(scores)
        best_neighbor = neighbors[best_neighbor_idx]
        best_neighbor_score = scores[best_neighbor_idx]

        if best_neighbor_score < current_score:
            current_solution = best_neighbor
            current_score = best_neighbor_score
            if verbose:
                print(f"Iter {iteration}: Score = {current_score}")
        else:
            break  # No improvement => local minimum reached

    return current_solution, current_score


def iterative_local_search(objective_function, generate_neighbors, perturbation,
                           initial_solution, nc, nf, meters, lines, wires_per_feeder,
                           max_iterations=1000, max_ils_iterations=10, verbose=False):
    """
    Perform iterative local search with repeated perturbations and local searches.

    Parameters:
        objective_function: function
        generate_neighbors: function
        perturbation: function(solution) -> perturbed solution
        initial_solution: numpy array
        nc, nf, meters, lines, wires_per_feeder: problem parameters
        max_iterations: int (local search max iterations)
        max_ils_iterations: int (number of ILS iterations)
        verbose: bool

    Returns:
        best_solution, best_score
    """
    current_solution = initial_solution.copy()
    current_score = objective_function(
        solution=current_solution,
        nc=nc,
        nf=nf,
        meters=meters,
        lines=lines,
        wires_per_feeder=wires_per_feeder
    )
    best_solution = current_solution.copy()
    best_score = current_score

    for ils_iter in range(max_ils_iterations):
        if verbose:
            print(f"ILS iteration {ils_iter+1}/{max_ils_iterations}, current best score: {best_score:.4f}")

        # Local Search from current solution
        local_sol, local_score = local_search(
            objective_function=objective_function,
            initial_solution=current_solution,
            generate_neighbors=generate_neighbors,
            nc=nc,
            nf=nf,
            meters=meters,
            lines=lines,
            wires_per_feeder=wires_per_feeder,
            max_iterations=max_iterations,
            verbose=verbose
        )

        # Update best if improved
        if local_score < best_score:
            best_solution = local_sol.copy()
            best_score = local_score

        # Perturb the local optimum to escape
        current_solution = perturbation(local_sol)

    return best_solution, best_score
