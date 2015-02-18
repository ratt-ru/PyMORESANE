#!/usr/bin/env python

import os
from distutils.core import setup

setup(name='pymoresane',
      version='0.1',
      description='CUDA-accelerated implementation of the MORESANE deconvolution algorithm',
      author='JSKenyon',
      author_email='jonosken@gmail.com',
      url='https://github.com/ratt-ru/PyMORESANE',
      packages=['pymoresane'],
      requires=['numpy', 'scipy', 'pyfits', 'pycuda'],
      scripts=['pymoresane/bin/runsane'],
      )
