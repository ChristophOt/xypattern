import pytest

import numpy as np
from xypattern import Pattern
from xypattern.combine import find_overlap, find_scaling, scale_patterns, stitch_patterns


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


def test_scale_patterns():
    p1 = Pattern(np.array([1, 2, 3]), np.array([1, 1, 1]))
    p2 = Pattern(np.array([1, 2, 3]), np.array([2, 2, 2]))
    p2.scaling = 3
    p3 = Pattern(np.array([2, 3, 4]), np.array([4, 4, 2]))

    scale_patterns([p1, p2, p3])
    assert p1.scaling == 1
    assert p2.scaling == 0.5
    assert p3.scaling == 0.25


def test_scale_patterns_unsorted():
    p1 = Pattern(np.array([1, 2, 3]), np.array([1, 1, 1]), name="p1")
    p2 = Pattern(np.array([5, 6, 7]), np.array([2, 2, 2]), name="p2")
    p3 = Pattern(np.array([3, 4, 5]), np.array([4, 4, 4]), name="p3")

    scale_patterns([p1, p2, p3])
    assert p1.scaling == 1
    assert p2.scaling == 0.5
    assert p3.scaling == 0.25


def test_scale_pattern_unsorted_different_length():
    [p1, p2, p3, p4] = generate_unsorted_patterns()

    scale_patterns([p1, p2, p3, p4])
    assert p1.scaling == 1
    assert p2.scaling == 0.5
    assert p3.scaling == 0.25
    assert p4.scaling == 0.125


def generate_unsorted_patterns():
    p1 = Pattern(np.array([1, 2, 3]), np.array([1, 1, 1]), name="p1")
    p2 = Pattern(np.array([5, 6, 7]), np.array([2, 2, 2]), name="p2")
    p3 = Pattern(np.array([3, 4, 5, 6, 7, 8, 9]), np.array([4, 4, 4, 4, 4, 4, 4]), name="p3")
    p4 = Pattern(np.array([8, 9, 10, 11]), np.array([8, 8, 8, 8]), name="p4")
    return [p1, p2, p3, p4]


def test_stitch_patterns():
    [p1, p2, p3, p4] = generate_unsorted_patterns()

    scale_patterns([p1, p2, p3, p4])
    p = stitch_patterns([p1, p2, p3, p4])
    assert p.x[0] == 1
    assert p.x[-1] == 11
    assert p.y[0] == 1
    assert np.array_equal(p.y, [1] * 11)
    assert p.scaling == 1
    assert p.x.shape == (11,)
    assert p.y.shape == (11,)
    assert p.x.shape == p.y.shape
