import itertools
import numpy as np


def generate_all_connection_vectors(n_apartments=10, n_phases=3):
    """
    Generate all possible apartment-to-phase assignments.
    Each vector has length `n_apartments`, with values in {0, ..., n_phases-1}.
    
    Returns:
        generator of numpy arrays
    """
    return (np.array(p) for p in itertools.product(range(n_phases), repeat=n_apartments))

