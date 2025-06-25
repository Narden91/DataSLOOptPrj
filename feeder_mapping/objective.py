import numpy as np

def objective_function(solution, nc, nf, meters, lines):
    """
    Evaluates how well the 'solution' assigns meters to feeders.
    Args:
        solution: list/np.array of length nc (values in 0 to nf-1)
        nc: number of consumers
        nf: number of feeders
        meters: numpy array of shape (nc, nt)
        lines: numpy array of shape (nf, nt)

    Returns:
        float: loss/error score
    """
    estimated_lines = np.zeros_like(lines)

    for f in range(nf):
        indices = np.where(solution == f)[0]
        if len(indices) > 0:
            estimated_lines[f, :] = np.sum(meters[indices, :], axis=0)

    error = np.mean((estimated_lines - lines) ** 2)
    return error
