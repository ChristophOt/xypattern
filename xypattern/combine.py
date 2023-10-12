import numpy as np
from .pattern import Pattern


def find_overlap(p1: Pattern, p2: Pattern) -> tuple[float, float] | None:
    """
    Find the overlap in x between two patterns
    :param p1: pattern 1
    :param p2: pattern 2
    :return: tuple of x_min and x_max for the overlapping region or None if no overlap can be found
    """
    x_min = max(p1.x[0], p2.x[0])
    x_max = min(p1.x[-1], p2.x[-1])
    if x_min > x_max:
        return None
    return x_min, x_max


def find_scaling(p1: Pattern, p2: Pattern) -> float:
    """
    Find the scaling factor of p2 to p1
    :param p1: pattern 1
    :param p2: pattern 2
    :return: scaling factor
    """
    overlap = find_overlap(p1, p2)
    p1_indices = np.where((p1.x >= overlap[0]) & (p1.x <= overlap[1]))
    p2_indices = np.where((p2.x >= overlap[0]) & (p2.x <= overlap[1]))
    return np.sum(p1.y[p1_indices]) / np.sum(p2.y[p2_indices])
