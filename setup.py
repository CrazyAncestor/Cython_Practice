from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension("test",
                                    sources=["test.pyx"],
                                    include_dirs=['./']),
                          annotate=True)
)
