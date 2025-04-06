#include "numpy/arrayobject.h"

static inline double
trapz_lib(double x_min, double x_max, double const *y, npy_int64 length)
{
    const npy_int64 n = length - 1;
    double accum = y[0];
    #pragma omp parallel for reduction(+ : accum) schedule(guided)
    for (npy_int64 i = 1; i < n; i++)
    {
        accum += 2.0 * y[i];
    }
    accum += y[n];
    return 0.5 * (x_max - x_min) / static_cast<double>(n) * accum;
}