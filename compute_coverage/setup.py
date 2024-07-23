from setuptools import setup, Extension, find_packages
import numpy

# Define the extension module
module = Extension('generate_coverage_vectors', sources=['src/gcc.c'], include_dirs=[numpy.get_include()])

# Run the setup
setup(
    name='GenerateCoverageVectors',
    version='0.1',
    description='Python Package with C extension for computing coverage vectors across a genome',
    # packages=find_packages(),
    ext_modules=[module]
)
