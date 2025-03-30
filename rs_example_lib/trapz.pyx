# cython: language_level = 3

cimport cython
cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "trapz.hpp" nogil:
    double trapz_lib(double x_min, double x_max, const double* y,
                     np.int64_t length)


@cython.boundscheck(False)
@cython.embedsignature(True)
@cython.wraparound(False)
def trapz_cython(double x_min, double x_max, double [::1] y):
    """Integrate with trapz method, in Cython."""
    cdef np.int64_t n = y.shape[0] - 1
    cdef double accum = y[0]
    cdef double accum2 = 0.0
    cdef np.int64_t i
    for i in range(1, n):
        accum2 += y[i]
    accum += (2.0 * accum2) + y[n]
    return 0.5 * (x_max - x_min) / n * accum


@cython.embedsignature(True)
def trapz_cpp(double x_min, double x_max, np.ndarray y):
    """Integrate with trapz method, in C++, in parallel."""
    cdef np.int64_t length = y.shape[0]
    cdef double result
    cdef double* y_p = <double*> np.PyArray_DATA(y)
    with nogil:
        result = trapz_lib(x_min, x_max, y_p, length)
    return result