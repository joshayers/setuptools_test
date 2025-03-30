# cython: language_level = 3

from libc cimport complex
from libc.math cimport M_PI
from numpy cimport int64_t
import cython


@cython.boundscheck(True)
@cython.embedsignature(True)
@cython.wraparound(False)
cpdef frf(double [::1] mass,
          double [::1] damp,
          double [::1] stiff,
          double [::1] freq,
          double [::1, :] force,
          double complex [::1, :] output):
    """Calculate an FRF in Cython."""
    cdef int64_t n_modes = mass.shape[0]
    cdef int64_t n_freq = freq.shape[0]
    cdef int64_t n_force = force.shape[1]
    cdef int64_t mode_i, freq_i, force_i
    cdef double w, w2
    cdef double complex h
    cdef double h_real, h_imag
    output[:, :] = 0.0
    for force_i in range(n_force):
        for freq_i in range(n_freq):
            w = (2.0 * M_PI) * freq[freq_i]
            w2 = w * w
            for mode_i in range(n_modes):
                h_real = -mass[mode_i] * w2 + stiff[mode_i]
                h_imag = damp[mode_i] * w
                h = h_real + 1j * h_imag
                output[freq_i, force_i] += force[mode_i, force_i] / h
