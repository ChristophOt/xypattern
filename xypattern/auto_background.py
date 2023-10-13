from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .pattern import Pattern

try:
    from .util.smooth_bruckner import smooth_bruckner
except ImportError as e:
    print(e)
    print("Could not import the Cython version of smooth_bruckner. Using python implementation instead.")
    from .util.smooth_bruckner_py import smooth_bruckner


class AutoBackground:
    @abstractmethod
    def extract_background(self, pattern: Pattern):
        """
        Extracts the background from a pattern.
        :param pattern: pattern to extract the background from
        :return: background pattern
        """
        raise NotImplementedError


class SmoothBrucknerBackground(AutoBackground):
    """
    Performs a background subtraction using bruckner smoothing and a chebyshev polynomial.
    Standard parameters are found to be optimal for synchrotron XRD.
    :param smooth_width: width of the window in x-units used for bruckner smoothing
    :param iterations: number of iterations for the bruckner smoothing
    :param cheb_order: order of the fitted chebyshev polynomial
    """

    def __init__(self, smooth_width=0.1, iterations=50, cheb_order=50):
        self.smooth_width = smooth_width
        self.iterations = iterations
        self.cheb_order = cheb_order

    def extract_background(self, pattern: Pattern):
        """
        """
        x, y = pattern.data
        smooth_points = int((float(self.smooth_width) / (x[1] - x[0])))

        y_smooth = smooth_bruckner(y, abs(smooth_points), self.iterations)
        # get cheb input parameters
        x_cheb = 2. * (x - x[0]) / (x[-1] - x[0]) - 1.
        cheb_parameters = np.polynomial.chebyshev.chebfit(x_cheb, y_smooth, self.cheb_order)

        return np.polynomial.chebyshev.chebval(x_cheb, cheb_parameters)
