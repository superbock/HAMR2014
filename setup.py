#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

extensions = [Extension('ConductionTracking',
                        ['ConductionTracking.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'])]

setup(
    name='ConductionTracking',
    version='0.01',
	author='Sebastian Böck',
    ext_modules=extensions,
	cmdclass={'build_ext': build_ext})
