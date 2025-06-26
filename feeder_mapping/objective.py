import numpy as np

# def objective_function(solution, nc, nf, meters, lines):
#     """
#     Evaluates how well the 'solution' assigns meters to feeders.
#     Args:
#         solution: list/np.array of length nc (values in 0 to nf-1)
#         nc: number of consumers
#         nf: number of feeders
#         meters: numpy array of shape (nc, nt)
#         lines: numpy array of shape (nf, nt)

#     Returns:
#         float: loss/error score
#     """
#     estimated_lines = np.zeros_like(lines)

#     for f in range(nf):
#         indices = np.where(solution == f)[0]
#         if len(indices) > 0:
#             estimated_lines[f, :] = np.sum(meters[indices, :], axis=0)

#     error = np.mean((estimated_lines - lines) ** 2)
#     return error


def objective_function_squared_sum(solution: np.ndarray, nc: int, nf: int, meters: np.ndarray, lines: np.ndarray, wires_per_feeder: int = 3) -> float:
    """
    Evaluates how well the 'solution' of wire assignments matches the supply data.

    This function implements the formula:
    min sum_{l=1 to nf} sum_{t=1 to nt} ( y_l(t) - sum_{i: cable(w_i)=l} x_i(t) )^2

    Args:
        solution: 1D array of shape (nc,). Represents the wire assignment `w_i` for
                  each consumer `i`. Values are 0-indexed, from 0 to
                  (nf * wires_per_feeder - 1).
        nc: Number of consumers.
        nf: Number of feeders (cables/lines).
        meters: Consumption data for each consumer `x_i(t)`. Shape (nc, nt).
        lines: Supply data for each feeder `y_l(t)`. Shape (nf, nt).
        wires_per_feeder: The number of wires that constitute one feeder/cable.
                          Defaults to 3 for a typical 3-phase system.

    Returns:
        float: The Sum of Squared Errors (SSE) score.
    """
    # The `solution` vector contains wire assignments. We need to map these to
    # feeder/cable assignments. This implements the `cable(w_i) = l` logic.
    # Using 0-indexed integer division, e.g., wires 0,1,2 -> feeder 0; 3,4,5 -> feeder 1.
    feeder_assignments = solution // wires_per_feeder

    # Initialize the estimated supply for each feeder line based on the proposed solution.
    estimated_lines = np.zeros_like(lines)

    # For each feeder `l`, find all consumers `i` connected to it and sum their consumption.
    for f_idx in range(nf):
        # Get indices of consumers whose assigned wire maps to the current feeder.
        consumer_indices = np.where(feeder_assignments == f_idx)[0]

        # If consumers are assigned to this feeder, sum their meter readings over time.
        if len(consumer_indices) > 0:
            estimated_lines[f_idx, :] = np.sum(meters[consumer_indices, :], axis=0)

    # Calculate the error as the sum of squared differences, as per the formula.
    # This is the Sum of Squared Errors (SSE), not the Mean (MSE).
    error = np.sum((estimated_lines - lines) ** 2)

    return float(error)