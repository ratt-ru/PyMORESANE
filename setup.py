#!/usr/bin/env python

from distutils.core import setup
from pymoresane import __version__

setup(name='pymoresane',
      version=__version__,
      description='CUDA-accelerated implementation of the MORESANE deconvolution algorithm',
      author='JSKenyon',
      author_email='jonosken@gmail.com',
      url='https://github.com/ratt-ru/PyMORESANE',
      packages=['pymoresane'],
      requires=['numpy', 'scipy', 'pyfits', 'pycuda'],
      scripts=['pymoresane/bin/runsane'],
      )
