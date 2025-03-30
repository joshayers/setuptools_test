import math
import time

import numpy as np

from rs_example_lib.trapz import trapz_cpp, trapz_cython


def trapz_python(x_min, x_max, y):
    n = y.shape[0] - 1
    accum = y[0]
    for i in range(1, n):
        accum += 2.0 * y[i]
    accum += y[n]
    return 0.5 * (x_max - x_min) / n * accum


def trapz_numpy(x_min, x_max, y):
    n = y.shape[0] - 1
    accum = y[0]
    accum += np.sum(y[1:n]) * 2.0
    accum += y[n]
    return 0.5 * (x_max - x_min) / n * accum


def time_trapz():
    x_min = 0.0
    x_max = 10000.0
    length = 50000000
    x = np.linspace(x_min, x_max, length)
    y = 1.0 + np.sin(20 * x) + np.sin(35 * x)

    results = []
    time_python = []
    for i in range(3):
        start = time.perf_counter()
        result = trapz_python(x_min, x_max, y)
        end = time.perf_counter()
        results.append(result)
        time_python.append(end - start)
    print(f"Python: {min(time_python):.4f}")

    time_numpy = []
    for i in range(3):
        start = time.perf_counter()
        result = trapz_numpy(x_min, x_max, y)
        end = time.perf_counter()
        results.append(result)
        time_numpy.append(end - start)
    print(f"NumPy: {min(time_numpy):.4f}")

    time_cython = []
    for i in range(3):
        start = time.perf_counter()
        result = trapz_cython(x_min, x_max, y)
        end = time.perf_counter()
        results.append(result)
        time_cython.append(end - start)
    print(f"Cython: {min(time_cython):.4f}")

    time_cpp = []
    for i in range(3):
        start = time.perf_counter()
        result = trapz_cpp(x_min, x_max, y)
        end = time.perf_counter()
        results.append(result)
        time_cpp.append(end - start)
    print(f"C++: {min(time_cpp):.4f}")

    result_1 = results[0]
    for result in results[1:]:
        assert math.isclose(result_1, result)


if __name__ == "__main__":
    time_trapz()
