import numpy as np
from pytest import approx
from xypattern.util.smooth_bruckner import smooth_bruckner
from xypattern.util.smooth_bruckner_py import smooth_bruckner as smooth_bruckner_py
from xypattern import Pattern
from .test_util import generate_peak_pattern


def test_both_smooth_bruckner_implementations_give_same_result():
    pattern = generate_peak_pattern()
    bkg1 = smooth_bruckner(pattern.y, 3, 50)
    bkg2 = smooth_bruckner_py(pattern.y, 3, 50)

    assert np.sum(bkg1 - bkg2) == approx(0.0, abs=1e-11)
