import numpy as np
from xypattern import Pattern


def gaussian(x, mu, a, sig):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def generate_peak_pattern(with_bkg=False):
    x = np.arange(0, 24, 0.01)
    y = np.zeros(x.shape)

    peaks = [
        [10, 3, 0.1],
        [12, 4, 0.1],
        [15, 6, 0.1],
    ]
    for peak in peaks:
        y += gaussian(x, peak[0], peak[1], peak[2])
    y_bkg = x * 0.4 + 5.0
    y_measurement = y + y_bkg

    if with_bkg:
        return Pattern(x, y_measurement), y_bkg
    return Pattern(x, y_measurement)
