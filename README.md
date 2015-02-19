# PyMORESANE

[![Join the chat at https://gitter.im/ratt-ru/PyMORESANE](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ratt-ru/PyMORESANE?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is a Python and pyCUDA-accelerated implementation of the MORESANE deconvolution algorithm (see [Dabbech et al. 2014](http://arxiv.org/abs/1412.5387) for a full description of the algorithm, and 
[Dabbech et al. 2012](http://www.academia.edu/1942933/Astronomical_image_deconvolution_using_sparse_priors_An_analysis-by-synthesis_approach) for earlier work).

### BASIC INSTALLATION

Install [numpy and scipy](http://www.scipy.org/install.html). 
Install [pyfits](http://www.stsci.edu/institute/software_hardware/pyfits/Download).

The following assumes you have numpy, scipy and pyfits installed.

Simply download all the files/zip from GitHub.

Place all files in a single directory, e.g. ~/PyMORESANE

Add the following to your .bashrc or .bash_aliases:  
alias runsane='python ~/PyMORESANE/pymoresane.py'

If you have put the PyMORESANE folder elsewhere, simply amend the alias to reflect its location.

Running pymoresane.py is as simple as typing runsane followed by the inputs you desire. runsane --help should provide some assistance, otherwise the docstrings/comments in the code are very detailed.

Feel free to contact me at jonosken@gmail.com with comments, questions, criticism and bug-reports. 




