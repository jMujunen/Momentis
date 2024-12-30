from Cython.Build import cythonize
from setuptools import Extension, setup
import numpy

# setup(ext_modules=cythonize(["**/*.pyx"]))
# setup(ext_modules=cythonize(["_FrameBuffer.pyx", "momentis.pyx"]))
extensions = [
    Extension("*", ["**/*.pyx"], include_dirs=[numpy.get_include()]),
]
setup(
    name="Momentis",
    ext_modules=cythonize(extensions),
)
