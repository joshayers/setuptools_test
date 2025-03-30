import math
import time

import numpy as np

from rs_example_lib.frf import frf

n_modes = 64 * 8
n_freq = 500 * 8
n_loops = 10


def pyy(freq, m, b, k, force_i):
    force = np.empty((n_modes, n_freq))
    for i in range(n_freq):
        force[:, i] = force_i[:, 0]

    Omega = 2 * np.pi * freq[None, :]
    H = (1j * b)[:, None] @ Omega + k[:, None] - m[:, None] @ Omega**2
    d = (force / H).sum(axis=0)
    return d


def new(freq, m, b, k, force):
    Omega = 2 * np.pi * freq[None, :]
    H = (1j * b)[:, None] @ Omega + k[:, None] - m[:, None] @ Omega**2
    d = (force / H).sum(axis=0)
    return d


def new4(freq, m, b, k, force):
    Omega = 2.0 * np.pi * freq
    H = np.empty((n_modes, n_freq), "c16", order="F")
    np.outer(b, Omega, out=H.imag)
    Omega *= Omega
    np.outer(-m, Omega, out=H.real)
    H.real += k[:, None]
    d = (force / H).sum(axis=0)
    return d


def new5(freq, m, b, k, force):
    Omega = 2.0 * np.pi * freq
    H = np.empty((n_freq, n_modes), "c16", order="C")
    np.outer(Omega, b, out=H.imag)
    Omega *= Omega
    np.outer(Omega, -m, out=H.real)
    H.real += k[None, :]
    d = (force.T / H).sum(axis=1)
    return d


def py_frf(mass, damp, stiff, freq, force, output):
    """Calculate an FRF."""
    n_modes = mass.shape[0]
    n_freq = freq.shape[0]
    n_force = force.shape[1]
    output[:, :] = 0.0
    for force_i in range(n_force):
        for freq_i in range(n_freq):
            w = (2.0 * math.pi) * freq[freq_i]
            w2 = w * w
            for mode_i in range(n_modes):
                h_real = -mass[mode_i] * w2 + stiff[mode_i]
                h_imag = damp[mode_i] * w
                h = h_real + 1j * h_imag
                output[freq_i, force_i] += force[mode_i, force_i] / h


def main():
    freq = np.linspace(1, 100, n_freq)
    m = np.random.random((n_modes,))
    b = np.random.random((n_modes,))
    k = np.random.random((n_modes,))
    force = np.asfortranarray(np.random.random((n_modes, 1)))
    force2 = np.asfortranarray(np.random.random((n_modes, 10 * 8)))
    force2[:, 0] = force[:, 0]

    times = []
    for i in range(n_loops):
        start = time.perf_counter()
        d_pyy = pyy(freq, m, b, k, force)
        end = time.perf_counter()
        times.append(end - start)
    print(f"PyYeti {min(times):.4f}")

    times = []
    for i in range(n_loops):
        start = time.perf_counter()
        d_new = new(freq, m, b, k, force)
        end = time.perf_counter()
        times.append(end - start)
    print(f"1d force {min(times):.4f}")

    times = []
    for i in range(n_loops):
        start = time.perf_counter()
        d_new4 = new4(freq, m, b, k, force)
        end = time.perf_counter()
        times.append(end - start)
    print(f"F-order H {min(times):.4f}")

    times = []
    for i in range(n_loops):
        start = time.perf_counter()
        d_new5 = new5(freq, m, b, k, force)
        end = time.perf_counter()
        times.append(end - start)
    print(f"C-order H {min(times):.4f}")

    d_pyt = np.empty((n_freq, force.shape[1]), "c16", order="F")
    times = []
    for i in range(n_loops * 0 + 1):
        start = time.perf_counter()
        py_frf(m, b, k, freq, force, d_pyt)
        end = time.perf_counter()
        times.append(end - start)
    print(f"Pure Python {min(times):.4f}")

    d_cyt = np.empty((n_freq, force.shape[1]), "c16", order="F")
    times = []
    for i in range(n_loops):
        start = time.perf_counter()
        frf(m, b, k, freq, force, d_cyt)
        end = time.perf_counter()
        times.append(end - start)
    print(f"Cython {min(times):.4f}")

    d_cyt3 = np.empty((n_freq, force2.shape[1]), "c16", order="F")
    times = []
    for i in range(n_loops):
        start = time.perf_counter()
        frf(m, b, k, freq, force2, d_cyt3)
        end = time.perf_counter()
        times.append(end - start)
    print(f"Cython 80x forces {min(times):.4f}")

    assert d_pyy.shape == (n_freq,)
    assert d_new.shape == (n_freq,)
    assert d_new4.shape == (n_freq,)
    assert d_new4.shape == (n_freq,)
    assert d_pyt.shape == (n_freq, 1)
    assert d_cyt.shape == (n_freq, 1)
    assert d_cyt3.shape == (n_freq, 10 * 8)
    np.testing.assert_allclose(d_pyy, d_new)
    np.testing.assert_allclose(d_pyy, d_new4)
    np.testing.assert_allclose(d_pyy, d_new5)
    np.testing.assert_allclose(d_pyy, d_pyt[:, 0])
    np.testing.assert_allclose(d_pyy, d_cyt[:, 0])
    np.testing.assert_allclose(d_pyy, d_cyt3[:, 0])


if __name__ == "__main__":
    main()
