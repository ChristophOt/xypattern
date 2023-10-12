import pytest

import numpy as np
from xypattern import Pattern
from xypattern.combine import find_overlap, find_scaling


def test_find_overlap():
    p1 = Pattern([1, 2, 3], [1, 2, 3])
    p2 = Pattern([2, 3, 4], [2, 3, 4])
    assert find_overlap(p1, p2) == (2, 3)

    p3 = Pattern([4, 5, 6], [4, 5, 6])
    assert find_overlap(p1, p3) is None


def test_find_scaling():
    p1 = Pattern(np.array([1, 2, 3]), np.array([1, 1, 1]))
    p2 = Pattern(np.array([1, 2, 3]), np.array([2, 2, 2]))
    assert find_scaling(p1, p2) == 0.5

    p3 = Pattern(np.array([2, 3, 4]), np.array([4, 4, 2]))
    assert find_scaling(p1, p3) == 0.25
