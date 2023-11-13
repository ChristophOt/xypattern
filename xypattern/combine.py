from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from .pattern import Pattern


def stitch_patterns(patterns: list[Pattern], binning=None) -> Pattern:
    """
    Stitch together a list of patterns.
    :param patterns: list of patterns to be stitched together
    :param binning: binning to be applied to the stitched pattern, if None, the binning of the first pattern will be
                    used
    :return: stitched pattern
    """
    if binning is None:
        binning = patterns[0].x[1] - patterns[0].x[0]

    x = np.concatenate([pattern.x for pattern in patterns])
    y = np.concatenate([pattern.y for pattern in patterns])
    return Pattern(x, y).rebin(binning)


def scale_patterns(patterns: list[Pattern]):
    """
    Scales a list of patterns to the first pattern in respect to x.  The scaling will be in place setting the scale
    attribute of the patterns.
    :param patterns: list of patterns to be scaled
    """
    for pattern in patterns:
        pattern.scaling = 1

    sorted_patterns = sorted(patterns, key=lambda p: p.x[0])
    for ind, pattern in enumerate(sorted_patterns):
        if ind == 0:
            pattern.scaling = 1
            continue

        scale_ind = ind - 1
        scaling = find_scaling(sorted_patterns[scale_ind], pattern)
        while scaling is None:
            scale_ind -= 1
            if scale_ind < 0:
                raise ValueError("No overlap found between patterns")
            scaling = find_scaling(sorted_patterns[scale_ind], pattern)

        pattern.scaling = scaling


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


def find_scaling(p1: Pattern, p2: Pattern) -> float | None:
    """
    Find the scaling factor of p2 to p1
    :param p1: pattern 1
    :param p2: pattern 2
    :return: scaling factor
    """
    overlap = find_overlap(p1, p2)
    if overlap is None:
        return None

    p1_indices = np.where((p1.x >= overlap[0]) & (p1.x <= overlap[1]))
    p2_indices = np.where((p2.x >= overlap[0]) & (p2.x <= overlap[1]))
    x1 = p1.x[p1_indices]
    x2 = p2.x[p2_indices]
    y1 = p1.y[p1_indices]
    y2 = p2.y[p2_indices]

    if len(x1) == len(x2) and np.allclose(x1, x2):
        return np.mean(y1 / y2)

    f2 = interp1d(x2, y2, kind="linear", fill_value="extrapolate")
    p2_interpolated = f2(x1)
    return np.mean(y1 / p2_interpolated)
