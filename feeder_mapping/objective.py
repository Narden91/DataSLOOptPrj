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



def objective_function_wire_assignment(solution: np.ndarray, nc: int, nf: int, 
                                     meters: np.ndarray, lines: np.ndarray, 
                                     wires_per_feeder: int = 3) -> float:
    """
    Evaluates how well the 'solution' of wire assignments matches the supply data.
    This function implements the mathematical formulation:
    
    f(w) = sum_{w=1 to nw} ||y_w - sum_{i=1 to nc} 1(w_i = w) * x_i||_2^2
    
    where:
    - w is the wire assignment vector
    - y_w is the measured supply for wire w
    - x_i is the consumption of apartment i
    - 1(w_i = w) is the indicator function (1 if apartment i is on wire w, 0 otherwise)
    - ||·||_2^2 is the squared Euclidean norm (sum of squared differences over time)
    
    Args:
        solution: 1D array of shape (nc,). Represents the wire assignment w_i for
                  each consumer i. Values are 0-indexed, from 0 to (nf * wires_per_feeder - 1).
                  This corresponds to w_i in {1,...,24} but using 0-based indexing.
        nc: Number of consumers (apartments). Typically 100.
        nf: Number of feeders (cables). Used to calculate total wires = nf * wires_per_feeder.
        meters: Consumption data for each consumer x_i(t). Shape (nc, nt).
                Each row represents one apartment's consumption over time.
        lines: Supply data for each wire y_w(t). Shape (nf * wires_per_feeder, nt).
               Each row represents one wire's measured supply over time.
        wires_per_feeder: Number of wires per feeder. Defaults to 3 for 3-phase systems.
    
    Returns:
        float: The Sum of Squared Errors (SSE) score representing the mismatch
               between predicted and actual wire supplies.
    
    Example:
        >>> nc, nf, wires_per_feeder, nt = 100, 8, 3, 720
        >>> nw = nf * wires_per_feeder  # 24 total wires
        >>> solution = np.random.randint(0, nw, size=nc)  # Random wire assignments
        >>> meters = np.random.rand(nc, nt)  # Consumer consumption data
        >>> lines = np.random.rand(nw, nt)   # Wire supply measurements
        >>> error = objective_function_wire_assignment(solution, nc, nf, meters, lines, wires_per_feeder)
    """
    # Calculate total number of wires
    nw = nf * wires_per_feeder
    
    # Validate input dimensions
    if solution.shape[0] != nc:
        raise ValueError(f"Solution must have {nc} elements, got {solution.shape[0]}")
    if meters.shape[0] != nc:
        raise ValueError(f"Meters must have {nc} rows, got {meters.shape[0]}")
    if lines.shape[0] != nw:
        raise ValueError(f"Lines must have {nw} rows, got {lines.shape[0]}")
    if meters.shape[1] != lines.shape[1]:
        raise ValueError("Meters and lines must have same number of time steps")
    
    # Create a boolean mask for wire assignments
    # assignment_mask[w, i] = True if consumer i is assigned to wire w
    assignment_mask = solution[np.newaxis, :] == np.arange(nw)[:, np.newaxis]
    
    # Calculate estimated wire supplies using matrix multiplication
    # This efficiently computes: sum_{i: w_i = w} x_i for all wires w
    estimated_wires = assignment_mask @ meters
    
    # Calculate the sum of squared errors across all wires and time steps
    error = np.sum((lines - estimated_wires) ** 2)
    
    return float(error)


def objective_function_correlation_based(solution: np.ndarray, nc: int, nf: int, 
                                       meters: np.ndarray, lines: np.ndarray, 
                                       wires_per_feeder: int = 3) -> float:
    """
    Alternative objective function based on correlation analysis.
    
    This function maximizes the correlation between predicted and actual wire supplies
    by minimizing the negative correlation coefficient. The intuition is that if 
    apartments are correctly assigned to wires, the temporal patterns of aggregated 
    consumption should strongly correlate with the measured wire supplies.
    
    Mathematical formulation:
    f(w) = -sum_{w=1 to nw} corr(y_w, sum_{i: w_i = w} x_i)
    
    where corr(a, b) is the Pearson correlation coefficient between vectors a and b.
    
    Args:
        solution: 1D array of shape (nc,). Wire assignments for each consumer.
        nc: Number of consumers (apartments).
        nf: Number of feeders (cables).
        meters: Consumption data for each consumer. Shape (nc, nt).
        lines: Supply data for each wire. Shape (nf * wires_per_feeder, nt).
        wires_per_feeder: Number of wires per feeder.
    
    Returns:
        float: Negative sum of correlation coefficients (lower is better).
    """
    # Calculate total number of wires
    nw = nf * wires_per_feeder
    
    # Validate input dimensions
    if solution.shape[0] != nc:
        raise ValueError(f"Solution must have {nc} elements, got {solution.shape[0]}")
    if meters.shape[0] != nc:
        raise ValueError(f"Meters must have {nc} rows, got {meters.shape[0]}")
    if lines.shape[0] != nw:
        raise ValueError(f"Lines must have {nw} rows, got {lines.shape[0]}")
    if meters.shape[1] != lines.shape[1]:
        raise ValueError("Meters and lines must have same number of time steps")
    
    # Create assignment mask
    assignment_mask = solution[np.newaxis, :] == np.arange(nw)[:, np.newaxis]
    
    # Calculate estimated wire supplies
    estimated_wires = assignment_mask @ meters
    
    total_correlation = 0.0
    valid_wires = 0
    
    for w_idx in range(nw):
        actual = lines[w_idx, :]
        predicted = estimated_wires[w_idx, :]
        
        # Skip wires with no assigned consumers or constant values
        if np.std(predicted) > 1e-10 and np.std(actual) > 1e-10:
            # Calculate Pearson correlation coefficient
            correlation = np.corrcoef(actual, predicted)[0, 1]
            # Handle NaN correlations (shouldn't happen with std check, but be safe)
            if not np.isnan(correlation):
                total_correlation += correlation
                valid_wires += 1
    
    # Return negative correlation (we want to minimize, so maximize correlation)
    # Normalize by number of valid wires to make scores comparable across different assignments
    if valid_wires > 0:
        return float(-total_correlation / valid_wires)
    else:
        return float(0.0)  # No valid correlations, neutral score


def objective_function_mae_based(solution: np.ndarray, nc: int, nf: int, 
                                meters: np.ndarray, lines: np.ndarray, 
                                wires_per_feeder: int = 3) -> float:
    """
    Alternative objective function based on Mean Absolute Error (L1 norm).
    
    This function uses L1 norm instead of L2 norm, making it more robust to outliers
    in the measurement data. The L1 norm gives equal weight to all deviations,
    while L2 norm (squared errors) heavily penalizes large deviations.
    
    Mathematical formulation:
    f(w) = sum_{w=1 to nw} sum_{t=1 to T} |y_w(t) - sum_{i: w_i = w} x_i(t)|
    
    Args:
        solution: 1D array of shape (nc,). Wire assignments for each consumer.
        nc: Number of consumers (apartments).
        nf: Number of feeders (cables).
        meters: Consumption data for each consumer. Shape (nc, nt).
        lines: Supply data for each wire. Shape (nf * wires_per_feeder, nt).
        wires_per_feeder: Number of wires per feeder.
    
    Returns:
        float: Sum of absolute errors (MAE-based objective).
    """
    # Calculate total number of wires
    nw = nf * wires_per_feeder
    
    # Validate input dimensions
    if solution.shape[0] != nc:
        raise ValueError(f"Solution must have {nc} elements, got {solution.shape[0]}")
    if meters.shape[0] != nc:
        raise ValueError(f"Meters must have {nc} rows, got {meters.shape[0]}")
    if lines.shape[0] != nw:
        raise ValueError(f"Lines must have {nw} rows, got {lines.shape[0]}")
    if meters.shape[1] != lines.shape[1]:
        raise ValueError("Meters and lines must have same number of time steps")
    
    # Create assignment mask
    assignment_mask = solution[np.newaxis, :] == np.arange(nw)[:, np.newaxis]
    
    # Calculate estimated wire supplies
    estimated_wires = assignment_mask @ meters
    
    # Calculate the sum of absolute errors (L1 norm)
    error = np.sum(np.abs(lines - estimated_wires))
    
    return float(error)


def objective_function_max_error_based(solution: np.ndarray, nc: int, nf: int, 
                                      meters: np.ndarray, lines: np.ndarray, 
                                      wires_per_feeder: int = 3) -> float:
    """
    Alternative objective function based on maximum error (L∞ norm).
    
    This function minimizes the worst-case error across all wires and time steps.
    It's particularly useful when you want to ensure that no single wire has
    a very large prediction error, even if the overall average error is acceptable.
    
    Mathematical formulation:
    f(w) = max_{w,t} |y_w(t) - sum_{i: w_i = w} x_i(t)|
    
    Args:
        solution: 1D array of shape (nc,). Wire assignments for each consumer.
        nc: Number of consumers (apartments).
        nf: Number of feeders (cables).
        meters: Consumption data for each consumer. Shape (nc, nt).
        lines: Supply data for each wire. Shape (nf * wires_per_feeder, nt).
        wires_per_feeder: Number of wires per feeder.
    
    Returns:
        float: Maximum absolute error across all wires and time steps.
    """
    # Calculate total number of wires
    nw = nf * wires_per_feeder
    
    # Validate input dimensions
    if solution.shape[0] != nc:
        raise ValueError(f"Solution must have {nc} elements, got {solution.shape[0]}")
    if meters.shape[0] != nc:
        raise ValueError(f"Meters must have {nc} rows, got {meters.shape[0]}")
    if lines.shape[0] != nw:
        raise ValueError(f"Lines must have {nw} rows, got {lines.shape[0]}")
    if meters.shape[1] != lines.shape[1]:
        raise ValueError("Meters and lines must have same number of time steps")
    
    # Create assignment mask
    assignment_mask = solution[np.newaxis, :] == np.arange(nw)[:, np.newaxis]
    
    # Calculate estimated wire supplies
    estimated_wires = assignment_mask @ meters
    
    # Calculate the maximum absolute error (L∞ norm)
    max_error = np.max(np.abs(lines - estimated_wires))
    
    return float(max_error)


def objective_function_huber_loss(solution: np.ndarray, nc: int, nf: int, 
                                 meters: np.ndarray, lines: np.ndarray, 
                                 wires_per_feeder: int = 3, delta: float = 1.0) -> float:
    """
    Alternative objective function based on Huber loss.
    
    Huber loss combines the benefits of both L1 and L2 norms:
    - For small errors (|error| ≤ δ), it behaves like L2 (quadratic)
    - For large errors (|error| > δ), it behaves like L1 (linear)
    
    This makes it robust to outliers while still being smooth and differentiable.
    
    Mathematical formulation:
    f(w) = sum_{w=1 to nw} sum_{t=1 to T} huber_δ(y_w(t) - sum_{i: w_i = w} x_i(t))
    
    where huber_δ(a) = {
        0.5 * a²           if |a| ≤ δ
        δ * |a| - 0.5 * δ²  if |a| > δ
    }
    
    Args:
        solution: 1D array of shape (nc,). Wire assignments for each consumer.
        nc: Number of consumers (apartments).
        nf: Number of feeders (cables).
        meters: Consumption data for each consumer. Shape (nc, nt).
        lines: Supply data for each wire. Shape (nf * wires_per_feeder, nt).
        wires_per_feeder: Number of wires per feeder.
        delta: Threshold parameter for Huber loss transition.
    
    Returns:
        float: Sum of Huber losses.
    """
    # Calculate total number of wires
    nw = nf * wires_per_feeder
    
    # Validate input dimensions
    if solution.shape[0] != nc:
        raise ValueError(f"Solution must have {nc} elements, got {solution.shape[0]}")
    if meters.shape[0] != nc:
        raise ValueError(f"Meters must have {nc} rows, got {meters.shape[0]}")
    if lines.shape[0] != nw:
        raise ValueError(f"Lines must have {nw} rows, got {lines.shape[0]}")
    if meters.shape[1] != lines.shape[1]:
        raise ValueError("Meters and lines must have same number of time steps")
    
    # Create assignment mask
    assignment_mask = solution[np.newaxis, :] == np.arange(nw)[:, np.newaxis]
    
    # Calculate estimated wire supplies
    estimated_wires = assignment_mask @ meters
    
    # Calculate residuals
    residuals = lines - estimated_wires
    abs_residuals = np.abs(residuals)
    
    # Apply Huber loss function
    huber_loss = np.where(
        abs_residuals <= delta,
        0.5 * residuals**2,  # Quadratic for small errors
        delta * abs_residuals - 0.5 * delta**2  # Linear for large errors
    )
    
    total_loss = np.sum(huber_loss)
    
    return float(total_loss)