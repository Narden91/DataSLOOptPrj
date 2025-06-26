def remove_deduplicate_neighbors(neighbors):
    """
    Remove duplicate np.array solutions from a list.
    Returns a list of unique np.arrays.
    """
    seen = set()
    unique = []
    for sol in neighbors:
        key = tuple(sol)  # np.array is unhashable, so convert to tuple
        if key not in seen:
            seen.add(key)
            unique.append(sol)
    return unique

def insert_neighborhood(solution):
    """
    For each apartment, try assigning it to the wire used by another apartment.
    (Excludes current assignment — no-op.)

    Returns:
        List of neighbors (np.array)
    """
    neighbors = []
    nc = len(solution)
    for i in range(nc):
        for j in range(nc):
            if i != j and solution[i] != solution[j]:
                neighbor = solution.copy()
                neighbor[i] = solution[j]
                neighbors.append(neighbor)
    return remove_deduplicate_neighbors(neighbors)