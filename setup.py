"""Setup."""

import numpy as np
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="rs_example_lib.frf",
            sources=["rs_example_lib/frf.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION", None)],
            language="c++",
        ),
        Extension(
            name="rs_example_lib.rss",
            sources=["rs_example_lib/rss.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION", None)],
            language="c++",
        ),
        Extension(
            name="rs_example_lib.trapz",
            sources=["rs_example_lib/trapz.pyx"],
            include_dirs=[np.get_include(), "rs_example_lib/"],
            define_macros=[("NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION", None)],
            language="c++",
        ),
    ]
)
