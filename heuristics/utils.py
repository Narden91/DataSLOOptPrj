import numpy as np

def random_perturbation(solution, num_changes=3, nw=None):
    """
    Randomly change `num_changes` positions in the solution to a random wire.
    nw = number of wires (max value +1)
    """
    perturbed = solution.copy()
    length = len(solution)
    for _ in range(num_changes):
        idx = np.random.randint(0, length)
        perturbed[idx] = np.random.randint(0, nw)
    return perturbed