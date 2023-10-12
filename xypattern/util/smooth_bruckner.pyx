
import numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def smooth_bruckner(y, Py_ssize_t smooth_points, Py_ssize_t iterations):
    cdef Py_ssize_t j, i, window_size, n

    n = y.size
    window_size = smooth_points * 2 + 1

    cdef double[:] y_extended = np.empty(n + smooth_points + smooth_points)

    for j in range(0, smooth_points):
        y_extended[j] = y[0]
    for j in range(n):
        y_extended[smooth_points+j] = y[j]
    for j in range(smooth_points + n,n + smooth_points + smooth_points):
        y_extended[j] = y[n-1]

    for j in range(0, iterations):
        window_avg = sum(y_extended[0: 2 * smooth_points + 1]) / (2 * smooth_points + 1)
        for i in range(smooth_points, n-smooth_points-2):
            if y_extended[i] > window_avg:
                y_new = window_avg
                # updating central value in average (first bracket)
                # and shifting average by one index (second bracket)
                window_avg += ((window_avg - y_extended[i]) + (
                            y_extended[i + smooth_points + 1] - y_extended[i - smooth_points])) / window_size
                y_extended[i] = y_new
            else:
                # shifting average by one index
                window_avg += (y_extended[i + smooth_points + 1] - y_extended[i - smooth_points]) / window_size
    return y_extended[smooth_points:smooth_points + n]
