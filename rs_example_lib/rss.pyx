# cython: language_level = 3

from libc.math cimport sqrt

cimport cython
from cython.parallel import prange
from numpy cimport int64_t, ndarray, PyArray_DATA
import numpy as np

@cython.boundscheck(False)
@cython.embedsignature(True)
@cython.wraparound(False)
def rss_cython(double [::1] x, double [::1] y):
    """Calculate RSS in Cython."""
    cdef int64_t n = x.shape[0]
    assert y.shape[0] == n
    cdef ndarray out = np.empty(n, 'f8')
    cdef double* out_p = <double*> PyArray_DATA(out)

    cdef int64_t i
    cdef double xi, yi
    with nogil:
        for i in range(n):
            xi = x[i]
            yi = y[i]
            out_p[i] = sqrt(xi * xi + yi * yi)

    return out


@cython.boundscheck(False)
@cython.embedsignature(True)
@cython.wraparound(False)
def rss_par_cython(double [::1] x, double [::1] y):
    """Calculate RSS in Cython, in parallel."""
    cdef int64_t n = x.shape[0]
    assert y.shape[0] == n
    cdef ndarray out = np.empty(n, 'f8')
    cdef double* out_p = <double*> PyArray_DATA(out)

    cdef int64_t i
    cdef double xi, yi
    with nogil:
        for i in prange(n):
            xi = x[i]
            yi = y[i]
            out_p[i] = sqrt(xi * xi + yi * yi)

    return out