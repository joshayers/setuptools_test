import time

import numpy as np

from rs_example_lib.rss import rss_cython, rss_par_cython


def rss_numpy(x, y):
    return np.sqrt(x * x + y * y)


def time_rss():
    length = 100000000
    x = np.random.random(length)
    y = np.random.random(length)

    results = []
    time_python = []
    for i in range(3):
        start = time.perf_counter()
        result = rss_numpy(x, y)
        end = time.perf_counter()
        results.append(result)
        time_python.append(end - start)
    print(f"Numpy: {min(time_python):.4f}")

    time_cython = []
    for i in range(3):
        start = time.perf_counter()
        result = rss_cython(x, y)
        end = time.perf_counter()
        results.append(result)
        time_cython.append(end - start)
    print(f"Cython: {min(time_cython):.4f}")

    time_par_cython = []
    for i in range(3):
        start = time.perf_counter()
        result = rss_par_cython(x, y)
        end = time.perf_counter()
        results.append(result)
        time_par_cython.append(end - start)
    print(f"Cython, Parallel: {min(time_par_cython):.4f}")

    result_1 = results[0]
    for result in results[1:]:
        np.testing.assert_allclose(result, result_1)


if __name__ == "__main__":
    time_rss()
